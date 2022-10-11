<h1 align="center">
  <br>
 :poultry_leg: Beating DeepFood paper. Computer Vision Project. :hotdog: <br>
</h1>

<p align="center">
  <a href="#about-paper">About Paper</a> •
  <a href="#about-methods">About Methods</a> •
  <a href="#result">Result</a> •
  <a href="#feedback">Feedback</a>
</p>

## About Paper
:bookmark_tabs: Paper: https://www.researchgate.net/publication/304163308_DeepFood_Deep_Learning-Based_Food_Image_Recognition_for_Computer-Aided_Dietary_Assessment

Worldwide, in 2014, more than 1.9 billion adults, 18 years and older, were overweight. Of these, over 600 million were obese. Accurately documenting dietary caloric intake is crucial to manage weight loss, but also presents challenges because most of the current methods for dietary assessment must rely on memory to recall foods eaten. The ultimate goal of our research is to develop computer-aided technical solutions to enhance and improve the accuracy of current measurements of dietary intake. Our proposed system in this paper aims to improve the accuracy of dietary assessment by analyzing the food images captured by mobile devices (e.g., smartphone). The key technique innovation in this paper is the deep learning-based food image recognition algorithms. Substantial research has demonstrated that digital imaging accurately estimates dietary intake in many environments and it has many advantages over other methods. However, how to derive the food information (e.g., food type and portion size) from food image effectively and efficiently remains a challenging and open research problem. We propose a new Convolutional Neural Network (CNN)-based food image recognition algorithm to address this problem. We applied our proposed approach to two real-world food image data sets (UEC-256 and Food-101) and achieved impressive results. To the best of our knowledge, these results outperformed all other reported work using these two data sets. Our experiments have demonstrated that the proposed approach is a promising solution for addressing the food image recognition problem. Our future work includes further improving the performance of the algorithms and integrating our system into a real-world mobile and cloud computing-based system to enhance the accuracy of current measurements of dietary intake.


![pic1](https://www.researchgate.net/publication/304163308/figure/fig1/AS:375253764198400@1466478877722/Inception-Module.png)

![pic2](https://www.researchgate.net/publication/304163308/figure/fig2/AS:375253764198401@1466478877750/Module-connection.png)

![pic3](https://www.researchgate.net/publication/304163308/figure/tbl1/AS:669301463851032@1536585312814/Comparison-of-accuracy-of-proposed-approach-using-bounding-box-on-UEC-256.png)


## About Methods
:computer: In this notebook I'm trying to beat [DeepFood paper](https://www.researchgate.net/publication/304163308_DeepFood_Deep_Learning-Based_Food_Image_Recognition_for_Computer-Aided_Dietary_Assessment) and of course the original [Food101 paper](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). It has 101 food categories, with 101'000 images. <br> 
DeepFood is a 2016 paper which used a Convolutional Neural Network trained for 2-3 days to achieve 77.4% top-1 accuracy.

### Here is a plan of this work
* :hammer: Using TensorFlow Datasets to download and explore data
  ```python
  # Load the data
  (train_data, test_data), ds_info = tfds.load(name="food101", 
                                               split=["train", "validation"], 
                                               shuffle_files=True, 
                                               as_supervised=True, 
                                               with_info=True)
  ```
* :hammer: Creating preprocessing function for our data
  ```python
  # Make a function for preprocessing images
  def preprocess_img(image, label, img_shape=224):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape]) # reshape to img_shape
    return tf.cast(image, tf.float32), label # return (float32_image, label) tuple
  ```
* :hammer: Batching & preparing datasets for modelling (**making our datasets run fast**)
  ```python
  # Map preprocessing function to training data (and paralellize)
  train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
  # Shuffle train_data and turn it into batches and prefetch it (load it faster)
  train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

  # Map prepreprocessing function to test data
  test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
  # Turn test data into batches (don't need to shuffle)
  test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)
  ```
* :hammer: Creating modelling callbacks
  ```python
  # Load the data
  def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instand to store log files.

    Stores log files with the filepath:
      "dir_name/experiment_name/current_datetime/"

    Args:
      dir_name: target directory to store TensorBoard log files
      experiment_name: name of experiment directory (e.g. efficientnet_model_1)
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving TensorBoard log files to:" + log_dir)
    return tensorboard_callback
  ```
* :pencil2: Setting up **mixed precision training**
  ```python
  # Turn on mixed precision training
  from tensorflow.keras import mixed_precision
  mixed_precision.set_global_policy(policy="mixed_float16") # set global policy to mixed precision 
  ```
* :hammer: Building a feature extraction model 
  ```python
  from tensorflow.keras import layers
  from tensorflow.keras.layers.experimental import preprocessing

  # Create base model
  input_shape = (224, 224, 3)
  base_model = tf.keras.applications.EfficientNetB0(include_top=False)
  base_model.trainable = False # freeze base model layers

  # Create Functional model 
  inputs = layers.Input(shape=input_shape, name="input_layer")
  # Note: EfficientNetBX models have rescaling built-in but if your model didn't you could have a layer like below
  # x = preprocessing.Rescaling(1./255)(x)
  x = base_model(inputs, training=False) # set base_model to inference mode only
  x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
  x = layers.Dense(len(class_names))(x) # want one output neuron per class 
  # Separate activation of output layer so we can output float32 activations
  outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x) 
  model = tf.keras.Model(inputs, outputs)

  # Compile the model
  model.compile(loss="sparse_categorical_crossentropy", # Use sparse_categorical_crossentropy when labels are *not* one-hot
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
  ```
* :pencil2: Fine-tuning the feature extraction model 
  ```python
  base_model.trainable = True
  ```

First of all, we used a mixed presicion paradigm. It means that if our GPU compute capability coefficient is greater then 7 - we can apply mixed presicion to our training process. <br> 
Worked in a Colab. It allows to use powerful GPUs. <br> <br>

## Result
:white_check_mark: We beat a DF paper with 78,3% of accuracy using Fine-tuning!

## Feedback
:person_in_tuxedo: Feel free to send me feedback on [Telegram](https://t.me/boramorka). Feature requests are always welcome. 

:abacus: [Check my other projects.](https://github.com/boramorka)