import os
import json
import argparse

import numpy as np

from python_tsp.exact import solve_tsp_dynamic_programming


def main():
  description = 'Generates a route betwwen the cluster centers using Traveling Salesperson Problems'
  parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--dir', type=str, help='the directory to which outputs will be saved', required=True)
  args = parser.parse_args()

  with open(os.path.join(args.dir, "data", "imagelists", f"imagelist-{args.dir}.json")) as file:
    imagelist = json.load(file)

  with open(os.path.join(args.dir, "data", "hotspots", f"hotspot-{args.dir}.json")) as file:
    hotspots = json.load(file)

  with open(os.path.join(args.dir, "data", "layouts", f"umap3d-{args.dir}.json")) as file:
    umap3d = json.load(file)

  images = {}
  for idx, image in enumerate(imagelist["images"]):
      images[image] = idx

  centers = []
  for hotspot in hotspots:
    center_image = hotspot["img"]
    center_idx = images[center_image]
    center_pos = umap3d[center_idx]
    centers.append(center_pos)

  np_centers = np.array(centers)

  print(len(centers))

  # https://jaykmody.com/blog/distance-matrices-with-numpy/
  distance_matrix = np.empty([len(centers), len(centers)])
  print(distance_matrix.shape)
  for i in range(len(centers)):
    distance_matrix[i, :] = np.sqrt(np.sum((np_centers[i] - np_centers) ** 2, axis=1))

  permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
  print(permutation, distance)

  output = [centers[idx] for idx in permutation]
  print(output)
  with open(os.path.join(args.dir, "data", "hotspots", f"route-{args.dir}.json"), 'w') as file:
    json.dump(output, file)


if __name__ == '__main__':
  main()
