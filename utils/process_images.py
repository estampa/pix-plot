#!/usr/bin/python

'''
Generates all data required to create a PixPlot viewer.

Documentation:
https://github.com/YaleDHLab/pix-plot

Usage:
  python utils/process_images.py --image_files="data/*/*.jpg"

                      * * *
'''

from __future__ import division, print_function
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.manifold import TSNE
from multiprocessing import Pool
from six.moves import urllib
from os.path import join
from PIL import Image
from umap import UMAP
from math import ceil
from glob import glob
import tensorflow as tf
import numpy as np
import json
import os
import re
import sys
import tarfile
import psutil
import subprocess
import codecs

import matplotlib.pyplot as plt

# configure command line interface arguments
flags = tf.app.flags
flags.DEFINE_string('model_dir', '/tmp/imagenet', 'The location of downloaded imagenet model')
flags.DEFINE_string('image_files', '', 'A glob path of images to process')
flags.DEFINE_integer('clusters', 20, 'The number of clusters to display in the image browser')
flags.DEFINE_boolean('validate_images', False, 'Whether to validate images before processing')
flags.DEFINE_string('output_folder', 'output', 'The folder where output files will be stored')
flags.DEFINE_string('layout', 'umap', 'The layout method to use {umap|tsne}')
flags.DEFINE_integer('dimensions', 3, 'The number of dimensions of the output space')
FLAGS = flags.FLAGS


class PixPlot:
  def __init__(self, image_glob):
    print(' * writing PixPlot outputs with ' + str(FLAGS.clusters) +
      ' clusters for ' + str(len(image_glob)) +
      ' images to folder ' + FLAGS.output_folder)

    self.image_files = image_glob
    self.output_dir = FLAGS.output_folder
    self.sizes = [32, 64, 128]  # eliminats 16 i 128
    self.format = 'png'
    self.n_clusters = FLAGS.clusters
    self.errored_images = set()
    self.vector_files = []
    self.image_vectors = []
    self.latent_vectors = []
    self.method = FLAGS.layout
    self.rewrite_image_thumbs = False
    self.rewrite_image_vectors = False
    self.rewrite_atlas_files = True
    self.validate_inputs(FLAGS.validate_images)
    self.create_output_dirs()
    self.create_image_thumbs()
    self.create_image_vectors()
    self.load_image_vectors()
    # self.load_and_mix_image_vectors()
    # self.load_and_mix_image_vectors_and_umap()

    # self.build_model(self.image_vectors)

    self.write_json()
    self.create_atlas_files()
    print('Processed output for ' + \
      str(len(self.image_files) - len(self.errored_images)) + ' images')


  def validate_inputs(self, validate_files):
    '''
    Make sure the inputs are valid, and warn users if they're not
    '''
    # ensure the user provided enough input images
    if len(self.image_files) < self.n_clusters:
      print('Please provide >= ' + str(self.n_clusters) + ' images')
      print(str(len(self.image_files)) + ' images were provided')
      sys.exit()

    if not validate_files:
      print(' * skipping image validation')
      return

    # test whether each input image can be processed
    print(' * validating input files')
    invalid_files = []
    for i in self.image_files:
      try:
        cmd = get_magick_command('identify') + ' "' + i + '"'
        response = subprocess.check_output(cmd, shell=True)
      except Exception as exc:
        invalid_files.append(i)
    if invalid_files:
      message = '\n\nThe following files could not be processed:'
      message += '\n  ! ' + '\n  ! '.join(invalid_files) + '\n'
      message += 'Please remove these files and reprocess your images.'
      print(message)
      sys.exit()


  def create_output_dirs(self):
    '''
    Create each of the required output dirs
    '''
    dirs = ['image_vectors', 'atlas_files', 'thumbs']
    for i in dirs:
      ensure_dir_exists( join(self.output_dir, i) )
    # make subdirectories for each image thumb size
    for i in self.sizes:
      ensure_dir_exists( join(self.output_dir, 'thumbs', str(i) + 'px') )


  def create_image_thumbs(self):
    '''
    Create output thumbs in 32px, 64px, and 128px
    '''
    print(' * creating image thumbs')
    resize_args = []
    n_thumbs = len(self.image_files)
    for c, j in enumerate(self.image_files):
      sizes = []
      out_paths = []
      for i in sorted(self.sizes, key=int, reverse=True):
        out_dir = join(self.output_dir, 'thumbs', str(i) + 'px')
        out_path = join( out_dir, get_filename(j) + '.' + self.format )
        if os.path.exists(out_path) and not self.rewrite_image_thumbs:
          continue
        sizes.append(i)
        out_paths.append(out_path)
      if len(sizes) > 0:
        resize_args.append([j, c, n_thumbs, sizes, out_paths])

    pool = Pool()
    for result in pool.imap(resize_thumb, resize_args):
      if result:
        self.errored_images.add( get_filename(result) )


  def create_image_vectors(self):
    '''
    Create one image vector for each input file
    '''
    self.download_inception()
    self.create_tf_graph()

    print(' * creating image vectors')
    with tf.Session() as sess:
      for image_index, image in enumerate(self.image_files):
        try:
          print(' * processing image', image_index+1, 'of', len(self.image_files))
          outfile_name = os.path.basename(image) + '.npy'
          out_path = join(self.output_dir, 'image_vectors', outfile_name)
          if os.path.exists(out_path) and not self.rewrite_image_vectors:
            continue
          # save the penultimate inception tensor/layer of the current image
          with tf.gfile.FastGFile(image, 'rb') as f:
            data = {'DecodeJpeg/contents:0': f.read()}
            feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            feature_vector = np.squeeze( sess.run(feature_tensor, data) )
            np.save(out_path, feature_vector)
          # close the open files
          for open_file in psutil.Process().open_files():
            file_handler = getattr(open_file, 'fd')
            os.close(file_handler)
        except Exception as exc:
          self.errored_images.add( get_filename(image) )
          print(' * image', get_ascii_chars(image), 'hit a snag', exc)


  def download_inception(self):
    '''
    Download the inception model to FLAGS.model_dir
    '''
    print(' * verifying inception model availability')
    inception_path = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    dest_directory = FLAGS.model_dir
    ensure_dir_exists(dest_directory)
    filename = inception_path.split('/')[-1]
    filepath = join(dest_directory, filename)
    if not os.path.exists(filepath):
      def progress(count, block_size, total_size):
        percent = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, percent))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(inception_path, filepath, progress)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


  def create_tf_graph(self):
    '''
    Create a graph from the saved graph_def.pb
    '''
    print(' * creating tf graph')
    graph_path = join(FLAGS.model_dir, 'classify_image_graph_def.pb')
    with tf.gfile.FastGFile(graph_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name='')


  def get_image_positions(self):
    '''
    Create a embedding of the image vectors
    '''
    print(' * calculating 2D/3D image positions')
    model = self.build_model(self.image_vectors)
    return self.get_dimensional_image_positions(model)


  def load_image_vectors(self):
    '''
    Return all image vectors
    '''
    print(' * loading image vectors')
    self.vector_files = glob( join(self.output_dir, 'image_vectors', '*') )
    for c, i in enumerate(self.vector_files):
      self.image_vectors.append(np.load(i))
      print(' * loaded', c+1, 'of', len(self.vector_files), 'image vectors')


  def load_and_mix_image_vectors_and_umap(self):
    '''
    Return all image vectors
    '''
    print(' * loading image vectors')
    self.vector_files = glob( join(self.output_dir, 'image_vectors-inception', '*') )
    for c, i in enumerate(self.vector_files):
      vector_inception = np.load(i)
      self.image_vectors.append(vector_inception)

      filename = os.path.basename(i)
      vector_latent_path = join(self.output_dir, 'image_vectors-latent', filename)
      vector_latent = np.load(vector_latent_path)
      vector_latent = np.float32(vector_latent)

      self.latent_vectors.append(vector_latent)
      print(' * loaded', c+1, 'of', len(self.vector_files), 'image vectors')

    dim = FLAGS.dimensions
    FLAGS.dimensions = 3
    fit_model = self.build_model(self.image_vectors)
    FLAGS.dimensions = dim


    for c, i in enumerate(fit_model):
      vector_latent = self.latent_vectors[c]
      vector = np.concatenate((i, vector_latent))
      self.image_vectors[c] = vector


  def load_and_mix_image_vectors(self):
    '''
    Return all image vectors
    '''
    print(' * loading image vectors')
    self.vector_files = glob( join(self.output_dir, 'image_vectors-inception', '*') )
    for c, i in enumerate(self.vector_files):
      filename = os.path.basename(i)
      vector_latent_path = join(self.output_dir, 'image_vectors-latent', filename)

      vector_inception = np.load(i)
      vector_latent = np.load(vector_latent_path)
      vector_latent = np.float32(vector_latent)
      # print(type(vector_inception), vector_inception.dtype, vector_inception.shape)
      # print(type(vector_latent), vector_latent.dtype, vector_latent.shape)
      vector = np.concatenate((vector_inception, vector_latent))

      # vector = vector_latent[0:3]

      self.image_vectors.append(vector)
      print(' * loaded', c+1, 'of', len(self.vector_files), 'image vectors')


  def build_model(self, image_vectors):
    '''
    Build a 2d projection of the `image_vectors`
    '''
    print(' * building 2D projection')
    if self.method == 'tsne':
      model = TSNE(n_components=FLAGS.dimensions, random_state=0)
      np.set_printoptions(suppress=True)
      return model.fit_transform( np.array(image_vectors) )

    elif self.method == 'umap':
      # model = UMAP(n_neighbors=25, min_dist=0.00001, metric='correlation')
      model = UMAP(n_components=FLAGS.dimensions, n_neighbors=2000, min_dist=2, metric='euclidean', verbose=True)
      # model = UMAP(n_components=FLAGS.dimensions, n_neighbors=25, min_dist=0.00001, metric='correlation')

      # for n_neighbors in (2, 5, 10, 20, 50, 100, 200, 500):
      #   for min_dist in (0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 1):
      #     for metric in ('euclidean', 'correlation', 'cosine'):
      #       self.draw_umap(np.array(image_vectors), n_neighbors, min_dist, FLAGS.dimensions, metric)

      # for n_neighbors in (500, 1000, 2000):
      #   for min_dist in (0.5, 1, 2):
      #     for metric in ('euclidean', 'correlation', 'cosine'):
      #       self.draw_umap(np.array(image_vectors), n_neighbors, min_dist, FLAGS.dimensions, metric)

      return model.fit_transform( np.array(image_vectors) )

  def draw_umap(self, data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = UMAP(
      n_neighbors=n_neighbors,
      min_dist=min_dist,
      n_components=n_components,
      metric=metric
    )
    print(data.shape)
    u = fit.fit_transform(data)
    fig = plt.figure()
    if n_components == 1:
      ax = fig.add_subplot(111)
      ax.scatter(u[:, 0], range(len(u)), c=data)
    if n_components == 2:
      ax = fig.add_subplot(111)
      ax.scatter(u[:, 0], u[:, 1], c=data)
    if n_components == 3:
      ax = fig.add_subplot(111, projection='3d')
      ax.scatter(u[:, 0], u[:, 1], u[:, 2], s=1)

    fig_file = "umap-ng_%03d-md_%0.5f-m_%s.png" % (n_neighbors, min_dist, metric)
    fig.savefig(fig_file)
    plt.close(fig)



  def get_dimensional_image_positions(self, fit_model):
    '''
    Write a JSON file that indicates the 2d position of each image
    '''
    print(' * writing JSON file')
    image_positions = []
    center = np.mean(fit_model, axis=0)
    for c, i in enumerate(fit_model):
      img = get_filename(self.vector_files[c])
      if img in self.errored_images:
        continue

      filename, file_extension = os.path.splitext(img)
      thumb_path = join(self.output_dir, 'thumbs', '32px', filename + "." + self.format)
      with Image.open(thumb_path) as image:
        width, height = image.size
      # Add the image name, x offset, y offset
      if FLAGS.dimensions == 2:
        image_positions.append([
          os.path.splitext(os.path.basename(img))[0],
          int((i[0] - center[0]) * 25),
          int((i[1] - center[1]) * 25),
          width,
          height
        ])
      else:
        image_positions.append([
          os.path.splitext(os.path.basename(img))[0],
          int((i[0] - center[0]) * 25),
          int((i[1] - center[1]) * 25),
          int((i[2] - center[2]) * 25),
          width,
          height
        ])
    return image_positions


  def get_centroids(self):
    '''
    Use KMeans clustering to find n centroid images
    that represent the center of an image cluster
    '''
    print(' * calculating ' + str(self.n_clusters) + ' clusters')
    model = KMeans(n_clusters=self.n_clusters)
    X = np.array(self.image_vectors)
    fit_model = model.fit(X)
    centroids = fit_model.cluster_centers_
    # find the points closest to the cluster centroids
    closest, _ = pairwise_distances_argmin_min(centroids, X)
    centroid_paths = [self.vector_files[i] for i in closest]
    centroid_json = []
    for c, i in enumerate(centroid_paths):
      centroid_json.append({
        'img': get_filename(i),
        'label': 'Cluster ' + str(c+1)
      })
    return centroid_json


  def write_json(self):
    '''
    Write a JSON file with image positions, the number of atlas files
    in each size, and the centroids of the k means clusters
    '''
    print(' * writing main JSON plot data file')
    out_path = join(self.output_dir, 'plot_data.json')
    with open(out_path, 'w') as out:
      json.dump({
        'centroids': self.get_centroids(),
        'positions': self.get_image_positions(),
        'atlas_counts': self.get_atlas_counts(),
      }, out)


  def get_atlas_counts(self):
    file_count = len(self.vector_files)
    return {
      '32px': ceil( file_count / (64**2) ),
      '64px': ceil( file_count / (32**2) ),
      '128px': ceil( file_count / (16**2) )
    }


  def create_atlas_files(self):
    '''
    Create image atlas files in each required size
    '''
    print(' * creating atlas files')
    atlas_group_imgs = []
    for thumb_size in self.sizes:
      # identify the images for this atlas group
      atlas_thumbs = self.get_atlas_thumbs(thumb_size)
      atlas_group_imgs.append(len(atlas_thumbs))
      self.write_atlas_files(thumb_size, atlas_thumbs)
    # assert all image atlas files have the same number of images
    assert all(i == atlas_group_imgs[0] for i in atlas_group_imgs)


  def get_atlas_thumbs(self, thumb_size):
    thumbs = []
    thumb_dir = join(self.output_dir, 'thumbs', str(thumb_size) + 'px')
    with open(join(self.output_dir, 'plot_data.json')) as f:
      for i in json.load(f)['positions']:
        thumbs.append( join(thumb_dir, i[0] + '.' + self.format) )
    return thumbs


  def write_atlas_files(self, thumb_size, image_thumbs):
    '''
    Given a thumb_size (int) and image_thumbs [file_path],
    write the total number of required atlas files at this size
    '''
    if not self.rewrite_atlas_files:
      return

    # build a directory for the atlas files
    out_dir = join(self.output_dir, 'atlas_files', str(thumb_size) + 'px')
    ensure_dir_exists(out_dir)

    # specify number of columns in a 2048 x 2048px texture
    atlas_cols = 2048/thumb_size

    # subdivide the image thumbs into groups
    atlas_image_groups = subdivide(image_thumbs, atlas_cols**2)

    # generate a directory for images at this size if it doesn't exist
    for idx, atlas_images in enumerate(atlas_image_groups):
      print(' * creating atlas', idx + 1, 'at size', thumb_size)
      out_path = join(out_dir, 'atlas-' + str(idx) + '.' + self.format)
      # write a file containing a list of images for the current montage
      tmp_file_path = join(self.output_dir, 'images_to_montage.txt')
      with codecs.open(tmp_file_path, 'w', encoding='utf-8') as out:
        # python 2
        try:
          out.write('\n'.join(map('"{0}"'.decode('utf-8').format, atlas_images)))
        # python 3
        except AttributeError:
          out.write('\n'.join(map('"{0}"'.format, atlas_images)))

      # build the imagemagick command to montage the images
      cmd =  get_magick_command('montage') + ' @' + tmp_file_path + ' '
      cmd += '-background none '
      cmd += '-size ' + str(thumb_size) + 'x' + str(thumb_size) + ' '
      cmd += '-geometry ' + str(thumb_size) + 'x' + str(thumb_size) + '+0+0 '
      cmd += '-tile ' + str(atlas_cols) + 'x' + str(atlas_cols) + ' '
      cmd += '-quality 85 '
      cmd += '-sampling-factor 4:2:0 '
      cmd += '"' + out_path + '"'
      os.system(cmd)

    # delete the last images to montage file
    try:
      os.remove(tmp_file_path)
    except Exception:
      pass


def get_magick_command(cmd):
  '''
  Return the specified imagemagick command prefaced with magick if
  the user is on Windows
  '''
  if os.name == 'nt':
    return 'magick ' + cmd
  return cmd


def resize_thumb(args):
  '''
  Create a command line request to resize an image
  Images for all thumb sizes are created in a single call, chaining the resize steps
  '''
  img_path, idx, n_imgs, sizes, out_paths = args
  print(' * creating thumb', idx+1, 'of', n_imgs, 'at sizes', sizes)
  cmd =  get_magick_command('convert') + ' '
  cmd += '-define jpeg:size={' + str(sizes[0]) + 'x' + str(sizes[0]) + '} '
  cmd += '"' + img_path + '" '
  cmd += '-strip '
  cmd += '-background none '
  cmd += '-gravity center '
  for i in range(0, len(sizes)):
    cmd += '-resize "' + str(sizes[i]) + 'X' + str(sizes[i]) + '>" '
    if not i == len(sizes)-1:
      cmd += "-write "
    cmd += '"' + out_paths[i] + '" '
  try:
    response = subprocess.check_output(cmd, shell=True)
    return None
  except subprocess.CalledProcessError as exc:
    return img_path


def subdivide(l, n):
  '''
  Return n-sized sublists from iterable l
  '''
  n = int(n)
  for i in range(0, len(l), n):
    yield l[i:i + n]


def get_ascii_chars(s):
  '''
  Return a string that contains the ascii characters from string `s`
  '''
  return ''.join(i for i in s if ord(i) < 128)


def get_filename(path):
  '''
  Return the root filename of `path` without file extension
  '''
  return os.path.splitext( os.path.basename(path) )[0]


def ensure_dir_exists(directory):
  '''
  Create the input directory if it doesn't exist
  '''
  if not os.path.exists(directory):
    os.makedirs(directory)


def limit_float(f):
  '''
  Limit the float point precision of float value f
  '''
  return int(f*10000)/10000


def main(*args, **kwargs):
  '''
  The main function to run
  '''
  # user specified glob path with tensorflow flags
  if FLAGS.image_files:
    image_glob = glob(FLAGS.image_files)
  # one argument was passed; assume it's a glob of image paths
  elif len(sys.argv) == 2:
    image_glob = glob(sys.argv[1])
  # many args were passed; assume the user passed a glob
  # path without quotes, and the shell auto-expanded them
  # into a list of file arguments
  elif len(sys.argv) > 2:
    image_glob = sys.argv[1:]

  # no glob path was specified
  else:
    print('Please specify a glob path of images to process\n' +
      'e.g. python utils/process_images.py "folder/*.jpg"')

  PixPlot(image_glob)

if __name__ == '__main__':
  tf.app.run()
