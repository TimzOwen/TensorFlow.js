import * as tf from ‘@tensorflow/tfjs’;
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [4], units: 100}));
model.add(tf.layers.dense({units: 4}));
model.compile({loss: ‘categoricalCrossentropy’, optimizer: ‘sgd’});
