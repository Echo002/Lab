import tensorflow as tf

inputs = tf.constant([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]],[[3],[2],[1]]])
print(inputs.get_shape())
print(inputs.get_shape().as_list()[1:])