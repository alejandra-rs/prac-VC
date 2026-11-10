# Práctica 4 - Visión por Computador
## [Marcial Galván](https://github.com/Jose-Marcial-GF) - [Alejandra Rodríguez](https://github.com/alejandra-rs)

poner requirements y como ejecutar tensorboard

## fuentes

[Modelo rt_detr y como usarlo](https://huggingface.co/docs/transformers/model_doc/rt_detr)

[Motivo por el que se usa el rtdetrv2](https://www.visionanalysis.org/articles/rtdetr-vs-rtdetrv2)

[Formato COCO rtdetrv2](https://roboflow.com/formats/coco-json)

[de donde he sacado el code](https://huggingface.co/docs/transformers/main/en/tasks/object_detection)


[Aquí se nombra el RT-detr_v2-L](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)

[Alternativa a mirar ECDet-L](https://www.visionanalysis.org/model/ec-l)

[Gráfica elección de modelos para comparar](https://www.visionanalysis.org/?families=ec%2Cyolo26%2Crtdetrv4%2Cdeimv2)

- RT-DETRv4-L: https://github.com/AIMIA-EAI/RT-DETRv4
- YOLO26-L:
- DEIMv2-L: https://github.com/Intellindust-AI-Lab/DEIM
- ECDet-L: https://github.com/Intellindust-AI-Lab/EdgeCrafter

[formato yolo](https://docs.ultralytics.com/datasets/detect)
[formato coco](https://cocodataset.org/#home)

## Elección de tracker
https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/multi-object-tracking-deepsort-bytetrack-ocsort

tras considerar la utilización de las siguientes familias:

https://cdn.prod.website-files.com/64e8910adc5a63966a68acea/6aca11f2ead14ac1b964c141_6aca11a7cf93033d9ed6f759_02_8-multi-object-tracking-deepsort-bytetrack-ocsort__02-four-tracker-comparison.svg

Se han evaluado los siguientes modelos byteTrack, que se basa en el algoritmo del kalman y permite asociar mediante IoU y la confianza.
Oc-sort está pensado para situaciones donde el objeto que se está siguiendo tiene cambios bruscos en la velocidad de movimiento o las imágenes son capturadas desde una cámara móvil
Bot-SORT-ReID permite reidentificar el objeto, ideal si un objeto desaparece de la escena un largo periodo de tiempo y cuando vuelva a aparecer se quiere reconocer, esto aumenta en el tiempo de cómputo.
Nuestro problema consiste en una cámara fija, a través de la que se quiere detectar y seguir vehículos, estos no tienen cambios bruscos de dirección ni velocidad, hace que no sea necesario utilizar OC-SORT.
Del mismo modo que no hay grandes periodos de oclusión de los vehículos ni personas por lo que no sería necesaro utilizar Bot-SORT-ReID para deidentificarlos.
Deforma que el algoritmo de kalman, en el que se basan ByteTrack y OC-SORT no debería ser una mala aproximación.
Sin embargo sí puede haber varios objetos en escena simultáneamente, en esos casos, evaluados por el benchmark de MoT Byte-track obtiene un mejor resultado que OC-SORT.
Es por eso que se ha utilizado Byte-Track como tacker
