import random

class QAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=0.2):
        # 每个键是 (状态, 动作)，每个值是对应的 Q 值。
        self.q_table = {}

        self.actions = list(actions)
        if len(self.actions) == 0:
            raise ValueError("至少需要提供一个动作")

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        # 先组合字典的键，再检查是否已有记录。
        key = (state, action)

        if key in self.q_table:
            q_value = self.q_table[key]
        else:
            q_value = 0.0

        return q_value

    def choose_action(self, state):
        """以 epsilon 的概率探索，否则选择当前 Q 值最大的动作。"""
        random_number = random.random()
        exploration_probability = self.epsilon

        if random_number < exploration_probability:
            # 探索：从动作列表中随机选择一个动作。
            action = random.choice(self.actions)
            return action

        # 利用：先把第一个动作当成当前最好的动作。
        # 不直接把最大值设为 0，因为所有动作的 Q 值可能都是负数。
        first_action = self.actions[0]
        best_value = self.get_q(state, first_action)
        best_actions = []
        best_actions.append(first_action)

        # 从第二个动作开始，逐个比较 Q 值。
        for action_index in range(1, len(self.actions)):
            action = self.actions[action_index]
            current_value = self.get_q(state, action)

            if current_value > best_value:
                best_value = current_value
                best_actions = []
                best_actions.append(action)
            elif current_value == best_value:
                best_actions.append(action)

        # 并列最优时随机选择，避免总是偏向列表中的第一个动作。
        selected_action = random.choice(best_actions)
        return selected_action
    
    def learn(self, state, action, reward, next_state, done):
        """根据 (状态, 动作, 奖励, 下一状态) 更新 Q 值。"""
        current_q = self.get_q(state, action)

        if done:
            target_q = reward
        else:
            # 计算下一状态的最大 Q 值。
            max_next_q = float("-inf")
            for next_action in self.actions:
                next_q = self.get_q(next_state, next_action)
                if next_q > max_next_q:
                    max_next_q = next_q

            target_q = reward + self.gamma * max_next_q

        # 根据 Q-learning 更新公式计算新的 Q 值。
        new_q = current_q + self.alpha * (target_q - current_q)

        # 更新 Q 表。
        key = (state, action)
        self.q_table[key] = new_q
