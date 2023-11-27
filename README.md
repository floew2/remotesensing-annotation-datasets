# Annotated, remote sensing based data sets for deep learning applications

The availability of annotated datasets is paramount for training and assessing machine learning algorithms geared towards the automated analysis of remote sensing data. Below is a non-comprehensive compilation of aerial and satellite imagery datasets featuring annotations for various computer vision and deep learning tasks, including instance segmentation, object detection, semantic segmentation, scene classification, and others. The subsequent table presents a review of several annotated remote sensing based datasets along with their respective characteristics.

## Overview annotated data sets

Scroll to the right to access the table. Down below this table, you will find an overview of annotation tools.

|                                            Name                                            |                               Topics                                |               Images/Image chips               |     N annotations      |                  Main Task                  | Classes | Publication year |  Image acquisition years   |                                               Regions                                                |                  Total area                  |                                         Publication Paper                                         | Storage (zipped) |               Size [px]               |                             Format                             |                                                     Download URL                                                      |                                       Example codes                                        | Pixel size [cm] |              Sensors               |     Type     |                                     Licence                                     |                       Licence URL                       |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|                               Oil Storage Detection (Sample)                               |           Oil Storage, POL (Petroleum, Oil and Lubricant)           |                       98                       |         13592          |              Object Detection               |    1    |       2021       |                            |                                                                                                      |                                              |                                                                                                   |       97MB       |               2560x2560               |                         Custom format                          |                   <https://www.kaggle.com/datasets/airbusgeo/airbus-oil-storage-detection-dataset>                    |                                                                                            |       120       |            SPOT6, SPOT7            |   Optical    |                                 CC BY-NC-SA 4.0                                 |  <https://creativecommons.org/licenses/by-nc-sa/4.0/>   |
|                                     Oil Storage Tanks                                      |                             Oil Storage                             |                     10000                      |                        |              Object Detection               |         |       2019       |                            |                                                                                                      |                                              |                                                                                                   |       4GB        |                512x512                |                              COCO                              |                          <https://www.kaggle.com/datasets/towardsentropy/oil-storage-tanks>                           |                                                                                            |                 | RGB images saved from Google Earth |   Optical    |                                                                                 |                                                         |
|                          AFO - Aerial dataset of floating objects                          |                                                                     |                      3647                      |         39991          |              Object Detection               |    6    |       2021       |                            |                                                                                                      |                                              |      <https://content.iospress.com/articles/integrated-computer-aided-engineering/ica210649>      |      4.4GB       |         1280x720 to 3840x2160         |                              YOLO                              |              <https://www.kaggle.com/datasets/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects>               | <https://www.kaggle.com/datasets/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects> |                 |        Aerial imagery, UAV?        |   Optical    |        Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License        |                                                         |
|                                Airbus Wind Turbines Patches                                |                            Wind Turbines                            |                                                |                        |              Object Detection               |    2    |       2021       |         2015-2017          |                                                                                                      |                                              |                                 <https://arxiv.org/abs/1411.4038>                                 |       1GB        |                128x128                |                        Only image chips                        |                       <https://www.kaggle.com/datasets/airbusgeo/airbus-wind-turbines-patches>                        |                                                                                            |       150       |            SPOT6, SPOT7            |   Optical    |                                 CC BY-NC-SA 4.0                                 |  <https://creativecommons.org/licenses/by-nc-sa/4.0/>   |
|               LandCover.ai - Woodlands, Water and Roads from Aerial Imagery                |                             Land Cover                              |                       41                       |                        |            Semantic Segmentation            |    4    |       2021       |                            |                                                Poland                                                |                  216.27 km2                  |                                <https://arxiv.org/abs/2005.02264>                                 |                  |                512x512                |                                                                |                                        <https://landcover.ai.linuxpolska.com/>                                        |                                                                                            |    25 to 50     |           Aerial imagery           |   Optical    | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License |   <http://creativecommons.org/licenses/by-nc-sa/4.0/>   |
| SolarDK: A high-resolution urban solar panel image classification and localization dataset |                          Solar Panels, PV                           | 23,417 hand labelled images for classification | 880 segmentation masks |  Image classification, Image Segmentation   |         |       2022       |            2021            |                                               Denmark                                                |                   3890 km2                   |                                <https://arxiv.org/abs/2212.01260>                                 |                  |                                       |                                                                |                                                <https://osf.io/aj539/>                                                |                                                                                            |    10 to 15     |                UAV                 |   Optical    |                       CC-By Attribution 4.0 International                       |                                                         |
|              AIDER (Aerial Image Dataset for Emergency Response Applications)              |                 Natural Disaster, Damage Assessment                 |                                                |          8540          |            Image classification             |    5    |       2020       |                            |                                                                                                      |                                              |                      <https://ieeexplore.ieee.org/abstract/document/9050881>                      |      263 MB      |                                       |                        Only image chips                        |                                         <https://zenodo.org/records/3888300>                                          |                         <https://github.com/ckyrkou/EmergencyNet>                          |                 |                UAV                 |   Optical    |               CC BY 4.0 LEGAL CODE Attribution 4.0 International                | <https://creativecommons.org/licenses/by/4.0/legalcode> |
|                            Satellite Images of Hurricane Damage                            |                 Natural Disaster, Damage Assessment                 |                     15000                      |         15000          |              Object Detection               |    2    |       2018       | before and after hurricane |                                                 USA                                                  |                                              |                                <https://arxiv.org/abs/1807.01688>                                 |      63 MB       |                128x128                | Image chips, file names represent centre coordinates of object | <https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized> |                        <https://github.com/qcao10/DamageDetection>                         |                 |                                    |   Optical    |                          Creative Commons Attribution                           |     <https://creativecommons.org/licenses/by/4.0/>      |
|                  MASATI dataset (v2) - MAritime SATellite Imagery dataset                  |                          Maritime vessels                           |                      7389                      |                        |              Object Detection               |    7    |       2018       |                            | different regions in Europe, Africa, Asia, the Mediterranean sea and the Atlantic and Pacific oceans |                                              |                             <https://www.iuii.ua.es/datasets/masati/>                             |      2.3GB       |                512x512                |                           PASCAL VOC                           |                                       <https://goo.gl/forms/xrBmFfbSuwpyb3wS2>                                        |                                                                                            |                 |       Microsoft® Bing™ Maps        |   Optical    |                                 Bing Maps terms                                 |   <https://www.microsoft.com/maps/product/terms.html>   |
|    Agriculture-Vision: A Large Aerial Image Database for Agricultural Pattern Analysis     |                        Agriculture, Parcels                         |                      3432                      |         94986          |                                             |    9    |                  |         2017-2020          |                                                 USA                                                  |                                              |                                <https://arxiv.org/abs/2001.01306>                                 |       20GB       |                512x512                |                         Custom format                          |                       <https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021>                       |                                                                                            |       20        |           Aerial imagery           |   Optical    |                                                                                 |                                                         |
|                                          FloodNet                                          |                          Flood, Land Cover                          |                      2343                      |          2343          |            Semantic Segmentation            |   10    |       2020       |            2017            |                                                 USA                                                  |                                              |                                <https://arxiv.org/abs/2012.02951>                                 |                  |                                       |                         Custom format                          |                <https://drive.google.com/drive/folders/1leN9eWVQcvWDVYwNb2GCo5ML_wBEycWD?usp=sharing>                 |                   <https://github.com/BinaLab/FloodNet-Supervised_v1.0>                    |                 |                UAV                 |   Optical    |           Community Data License Agreement -- Permissive, Version 1.0           |           <https://cdla.dev/permissive-1-0/>            |
|                                           PASTIS                                           |                        Agriculture, Parcels                         |                      2433                      |         124422         |            Semantic Segmentation            |   19    |       2021       |         2018-2019          |                                                France                                                |                   40000km2                   |                               <https://arxiv.org/abs/2112.07558v1>                                |      28.8GB      |                128×128                |                         Custom format                          |                                         <https://zenodo.org/records/5012942>                                          |                      <https://github.com/VSainteuf/pastis-benchmark>                       |      1000       |             Sentinel-2             |   Optical    |                              CC BY 4.0 LEGAL CODE                               | <https://creativecommons.org/licenses/by/4.0/legalcode> |
|                                           PASTIS                                           |                        Agriculture, Parcels                         |                      2433                      |         124422         |            Semantic Segmentation            |   19    |       2021       |         2018-2019          |                                                France                                                |                   40000km2                   |                               <https://arxiv.org/abs/2112.07558v1>                                |     53.7 GB      |                128×128                |                         Custom format                          |                                         <https://zenodo.org/records/5735646>                                          |                      <https://github.com/VSainteuf/pastis-benchmark>                       |      1000       |       Sentinel-1, Sentinel-2       | Optical, SAR |                              CC BY 4.0 LEGAL CODE                               | <https://creativecommons.org/licenses/by/4.0/legalcode> |
|                                         RarePlanes                                         |                              Aircrafts                              |         253 (real), 50000 (synthetic)          |         614700         |              Object Detection               |   110   |       2020       |                            |                    USA, China, Spain, France, Germany, Russia, Brazil, many more                     | 9331.2 km\^2 (synthetic), 2,142 km\^2 (real) |                                <https://arxiv.org/abs/2006.02963>                                 |     325632MB     | 512×512 (real), 1920×1080 (synthetic) |                                                                |                                       <https://www.cosmiqworks.org/rareplanes/>                                       |                        <https://github.com/jdc08161063/RarePlanes>                         |       30        |            WorldView-3             |   Optical    |                              CC-4.0-BY-SA license                               |    <https://creativecommons.org/licenses/by-sa/4.0/>    |
|                                            xDB                                             | Natural Disasters, Building Damage, Post Disaster Damage Assessment |                     850736                     |         850736         |              Object Detection               |    6    |       2019       |         2011-2019          |               USA, Indonesia, Nepal, Bangladesh, Portgal, Mexico, Australia, Guatemala               |                  45000 sqkm                  |                                <https://arxiv.org/abs/1911.09296>                                 |       51GB       |               1024x1024               |                         Custom format                          |                                             <https://xview2.org/dataset>                                              |                                                                                            |      30-80      | WorldView-2, WorldView-3, GeoEye1  |   Optical    |                                 CC BY-NC-SA 4.0                                 |  <https://creativecommons.org/licenses/by-nc-sa/4.0/>   |
|                                        SEN12-FLOOD                                         |                      Natural Disasters, Flood                       |                                                |                        | Image Classification, Semantic Segmentation |         |       2020       |         2018-2019          |                        West and South-East Africa, Middle-East, and Australia                        |                                              | <https://ieee-dataport.org/open-access/sen12-flood-sar-and-multispectral-dataset-flood-detection> |     12288MB      |                512x512                |                                                                |           <https://ieee-dataport.org/open-access/sen12-flood-sar-and-multispectral-dataset-flood-detection>           |                                                                                            |      1000       |       Sentinel-1, Sentinel-2       | Optical, SAR |                 CC BY 4.0 Deed - Attribution 4.0 International                  |     <https://creativecommons.org/licenses/by/4.0/>      |

## Overview annotation tools for geospatial data

xxxxx

| Tool name                         | Description                                                                                                     | Pricing                                               | URL                                                        | Examples                                              |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------- |
| Labelbox                         | Labelbox supports annotation of Geotiffs and provides annotations with geospatial coordinates.                | Free and commercial subscription fee versions available | [labelbox.com](https://labelbox.com/)                      | -                                                     |
| Groundwork (Azavea)              | An annotation tool for efficiently labeling geospatial data, including satellite imagery, to create training datasets for neural network models from satellite imagery. | Free and commercial subscription fee versions available | [groundwork.azavea.com](https://groundwork.azavea.com/)     | -                                                     |
| LabelStud.io                     | Craft comprehensive datasets tailored for the training of YOLO object detection models.                         | Open Source                                           | [labelstud.io](https://labelstud.io)                       | Annotations in YOLO                                   |
| Labelme                          | [github.com/wkentaro/labelme](https://github.com/wkentaro/labelme)                                              | Annotation with GeoTiffs                             | Convert to COCO                                           | -                                                     |
| Labelimg                         | [pypi.org/project/labelImg](https://pypi.org/project/labelImg/)                                                | -                                                     | -                                                          | -                                                     |
| Satellite Imagery Label Tool (Microsoft) | Utilize intuitive web interface to effortlessly create and share vector annotations across scenes of satellite/aerial imagery. | [github.com/microsoft/satellite-imagery-labeling-tool](https://github.com/microsoft/satellite-imagery-labeling-tool) | -                                                          | -                                                     |


xxxx

+----------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| Tool name                                                                                                | Description                                                                                                                                                             | Pricing                                                 | URL                                                            | Examples                                                                                                                       |
+==========================================================================================================+=========================================================================================================================================================================+=========================================================+================================================================+================================================================================================================================+
| [Labelbox](https://labelbox.com/)                                                                        | Labelbox supports annotation of Geotiffs and provides annotations with geospatial coordinates.                                                                          | Free and commercial subscription fee versions available | <https://labelbox.com/>                                        |                                                                                                                                |
+----------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| [Groundwork](https://groundwork.azavea.com/) (Azavea)                                                    | An annotation tool for efficiently labeling geospatial data, including satellite imagery, to create training datasets for neural network models from satellite imagery. | Free and commercial subscription fee versions available | <https://groundwork.azavea.com/>                               |                                                                                                                                |
+----------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| [LabelStud.io](https://labelstud.io)                                                                     | Craft comprehensive datasets tailored for the training of YOLO object detection models.                                                                                 | Open Source                                             | <https://labelstud.io>                                         | [Annotations in YOLO](https://labelstud.io/blog/quickly-create-datasets-for-training-yolo-object-detection-with-label-studio/) |
+----------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| [Labelme](https://pypi.org/project/labelme/)                                                             |                                                                                                                                                                         |                                                         | <https://github.com/wkentaro/labelme>                          | [Annotation with GeoTiffs](https://medium.com/@wvsharber/labelme-image-annotation-for-geotiffs-b460ba83804f)                   |
|                                                                                                          |                                                                                                                                                                         |                                                         |                                                                |                                                                                                                                |
|                                                                                                          |                                                                                                                                                                         |                                                         |                                                                | [Convert to COCO](https://github.com/fcakyon/labelme2coco)                                                                     |
+----------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| [Labelimg](https://pypi.org/project/labelImg/)                                                           |                                                                                                                                                                         |                                                         | <https://pypi.org/project/labelImg/>                           |                                                                                                                                |
+----------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| [Satellite Imagery Label Tool](https://github.com/microsoft/satellite-imagery-labeling-tool) (Microsoft) | Utilize intuitive web interface to effortlessly create and share vector annotations across scenes of satellite/aerial imagery.                                          |                                                         | <https://github.com/microsoft/satellite-imagery-labeling-tool> |                                                                                                                                |
+----------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+

## Overview freely available, high resolution remote sensing imagery

+--------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------+
| Source name              | Description                                                                                                                                                                                                                                                                                   | URL                               |
+==========================+===============================================================================================================================================================================================================================================================================================+===================================+
| OpenAerialMap            | OpenAerialMap comprises a suite of utilities designed for the exploration, dissemination, and utilization of openly licensed satellite and unmanned aerial vehicle (UAV) imagery. Check out their license.                                                                                    | <https://map.openaerialmap.org/>  |
+--------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------+
| Maxar Open Data Program  | In the event of any natural disaster, Maxar's Open Data Program provides satellite imagery to support relief efforts. This includes incidents such as fires, floods, hurricanes, typhoons, and earthquakes. The data is released into the public domain under a Creative Commons 4.0 license. | <https://www.maxar.com/open-data> |
+--------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------+
| Airbus Defence and Space | Airbus Defense Systems provides over 140 sample images specifically for experimentation purposes. Check out their license.                                                                                                                                                                    | <https://intelligence.airbus.com> |
+--------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------+
