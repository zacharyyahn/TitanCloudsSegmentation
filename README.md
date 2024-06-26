<h2>Instance Segmentation of Clouds on Titan</h2>

![Header Figure](/assets/iou_compare7.png)


This is the repository for our paper _Rapid Automated Mapping of Clouds on Titan with Instance Segmentation_

**Abstract** <br>
Despite widespread adoption of deep learning models to address a variety of computer vision tasks, planetary science has yet to see extensive utilization of such tools to address its unique problems. On Titan, the largest moon of Saturn, tracking seasonal trends and weather patterns of clouds provides crucial insights into one of the most complex climates in the Solar System, yet much of the available image data is still analyzed in a conventional way. In this work, we apply a Mask R-CNN trained via transfer learning to perform instance segmentation of clouds in Titan images acquired by the Cassini spacecraft - a previously unexplored approach to a `big data' problem in planetary science. We demonstrate that an automated technique can provide quantitative measures for clouds, such as areas and centroids, that may otherwise be prohibitively time-intensive to produce by human mapping. Furthermore, despite Titan-specific challenges, our approach yields accuracy comparable to contemporary cloud identification studies on Earth and other worlds. We compare the efficiencies of human-driven versus algorithmic approaches, showing that transfer learning provides speed-ups that may open new horizons for data investigation for Titan. Moreover, we suggest that such approaches have broad potential for application to similar problems in planetary science where they are currently under-utilized. Future planned missions to the planets and remote sensing initiatives for the Earth promise to provide a deluge of image data in the coming years that will benefit strongly from leveraging machine learning approaches to perform the analysis.

**Data and Model** <br>
Dataset and model are available at [https://zenodo.org/records/11657501]. 

Data should be placed in folder `Dataset/` at the same level as `SemanticSegmentation/` and `MetricCalculation/`. <br>
Model should be placed in folder `SemanticSegmentation/saved_models/`. <br>

**Code** <br>
`MetricCalculation/` contains code for calculating area, centroid, and aspect ratio for each cloud. <br>
`--calculate_metrics.py` computes metrics for each cloud and saves them in `Metrics.csv`.

`SemanticSegmentation/` contains code for tuning, training, and evaluating model. <br>
`--figures/` contains training log generated by `main.py`. <br>
`--models/` contains model definition code. <br>
`--saved_models/` should contain model checkpoint, also contains saved config file.<br>
`--utils/` additional helper code. <br>
`--dataset.py` defines dataset preprocessing. <br>
`--eval.py` evaluates model on the test set. <br>
`--hparams.yaml` config file for model training. <br>
`--main.py` runs model training. <br>
