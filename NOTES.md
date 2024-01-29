# notes

- Last year's competition (2023)
  - https://www.kaggle.com/competitions/geolifeclef-2023-lifeclef-2023-x-fgvc10/data

- Generating Tiles from raster data on local machine
  - Download MS4W GDAL Installer package https://ms4w.com/download.html#ms4w-base-package
  - Navigate into the folder with gdal2tiles.py C:\Program Files\ms4w\gdalbindings\python\gdal
  - Use the following command line prompt to use GDAL tiling service
  - Command line prompt: python gdal2tiles.py -z (zoom level range, ex: ‘12-18’) (Directory of .tif raster file) (Destination Directory)
