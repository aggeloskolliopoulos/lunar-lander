#!/usr/bin/env python3
import argparse
import random
import collections
import numpy as np

try:
    import gymnasium as gym
except Exception:
    import gym

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

def build_q_network(input_dim, n_actions):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dense(n_actions, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model

def select_action(model, state, n_actions, epsilon):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    q_values = model.predict(state[np.newaxis, :], verbose=0)[0]
    return int(np.argmax(q_values))

def train_step(q_net, target_q_net, optimizer, batch, gamma=0.99):
    states = np.array(batch.state, dtype=np.float32)
    actions = np.array(batch.action, dtype=np.int32)
    rewards = np.array(batch.reward, dtype=np.float32)
    next_states = np.array(batch.next_state, dtype=np.float32)
    dones = np.array(batch.done, dtype=np.float32)

    q_targets = q_net.predict(states, verbose=0)
    next_q = target_q_net.predict(next_states, verbose=0)
    max_next_q = np.max(next_q, axis=1)
    targets = rewards + (1.0 - dones) * gamma * max_next_q
    for i, a in enumerate(actions):
        q_targets[i, a] = targets[i]

    history = q_net.fit(states, q_targets, epochs=1, verbose=0, batch_size=64)
    return float(history.history["loss"][0])

def soft_update(target_net, source_net, tau=0.005):
    target_weights = target_net.get_weights()
    source_weights = source_net.get_weights()
    new_weights = [tw * (1 - tau) + sw * tau for tw, sw in zip(target_weights, source_weights)]
    target_net.set_weights(new_weights)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="LunarLander-v2")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--buffer_size", type=int, default=200_000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epsilon_start", type=float, default=1.0)
    ap.add_argument("--epsilon_end", type=float, default=0.05)
    ap.add_argument("--epsilon_decay", type=float, default=0.995)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    args = ap.parse_args()

    env = gym.make(args.env_id)
    reset_out = env.reset()
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    n_actions = env.action_space.n
    state_dim = len(state)

    q_net = build_q_network(state_dim, n_actions)
    target_q_net = build_q_network(state_dim, n_actions)
    target_q_net.set_weights(q_net.get_weights())

    buffer = ReplayBuffer(capacity=args.buffer_size)
    epsilon = args.epsilon_start

    all_rewards = []
    all_losses = []

    for ep in range(args.episodes):
        reset_out = env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        ep_reward = 0.0

        while not done:
            action = select_action(q_net, state, n_actions, epsilon)
            step_out = env.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_out
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            if len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                loss = train_step(q_net, target_q_net, q_net.optimizer, batch, gamma=args.gamma)
                all_losses.append(loss)
                soft_update(target_q_net, q_net, tau=args.tau)

        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        all_rewards.append(ep_reward)
        print(f"Episode {ep+1}/{args.episodes}  reward={ep_reward:.2f}  epsilon={epsilon:.3f}")

        if (ep + 1) % 50 == 0:
            q_net.save("dqn_lunar_lander_clean.keras")
            print("Checkpoint saved: dqn_lunar_lander_clean.keras")

    q_net.save("dqn_lunar_lander_clean.keras")
    print("Training complete. Saved model to dqn_lunar_lander_clean.keras")

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(all_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.tight_layout()
        plt.show()

        if all_losses:
            plt.figure()
            plt.plot(all_losses)
            plt.title("Training Loss (per update)")
            plt.xlabel("Update step")
            plt.ylabel("Loss")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print("Plotting failed:", e)

if __name__ == "__main__":
    main()
