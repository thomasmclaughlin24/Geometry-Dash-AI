import keras
import tensorflow as tf

def LinearQNet(input_size, hidden_size, output_size):
    inputs = keras.layers.Input(shape=input_size, name="Input")
    layer1 = keras.layers.Dense(hidden_size, activation="relu", name="Dense1")(inputs)
    action = keras.layers.Dense(output_size, name="Dense2")(layer1)
    return keras.Model(inputs=inputs, outputs=action)
class JumpTrainer():
    def __init__(self, model, learningrate=1e-4, gamma=0.9):
        self.model = model
        self.gamma = gamma
        self.optimizer = keras.optimizers.Adam(learning_rate=learningrate)
        self.loss_object = keras.losses.MeanSquaredError()

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        future_rewards = tf.reduce_max(self.model(next_states), axis=1)
        updated_q_values = rewards + tf.math.multiply(self.gamma, future_rewards)
        updated_q_values = tf.math.multiply(updated_q_values, (1 - dones))
        mask = actions
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_actions = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
            loss = self.loss_object(updated_q_values, q_actions)
            grad = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables)) 