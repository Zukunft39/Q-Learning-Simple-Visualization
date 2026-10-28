from Environment import GridWorld
from QAgent import QAgent


def main():
    env = GridWorld(mode="fixed", seed=42)
    actions = ["w", "a", "s", "d"]

    # 只创建一次智能体，测试时继续使用它已经学到的 Q 表。
    agent = QAgent(actions, epsilon=0.2)
    episode_count = 100
    test_episode_count = 100
    max_steps = 300

    print("开始训练：100 局，epsilon = 0.2")
    training_success_count = 0

    for episode_index in range(episode_count):
        state = env.reset()
        total_reward = 0
        reached_goal = False
        steps_used = 0

        for step_index in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            # 训练阶段根据每一步经验更新 Q 表。
            agent.learn(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state
            steps_used = step_index + 1

            if done:
                reached_goal = True
                training_success_count += 1
                break

        episode_number = episode_index + 1
        print(
            f"训练第 {episode_number} 局，"
            f"总奖励：{total_reward}，"
            f"到达终点：{reached_goal}，"
            f"步数：{steps_used}，"
            f"撞墙次数：{env.hit_wall_num}"
        )

    training_success_rate = training_success_count / episode_count * 100
    print(f"训练成功率：{training_success_rate:.1f}%")

    # 训练结束后关闭随机探索，不重新创建智能体，也不清空 Q 表。
    agent.epsilon = 0.0
    print("\n开始测试：100 局，epsilon = 0，不更新 Q 表")

    test_success_count = 0
    test_reward_sum = 0
    test_steps_sum = 0
    test_collisions_sum = 0

    for episode_index in range(test_episode_count):
        state = env.reset()
        total_reward = 0
        reached_goal = False
        steps_used = 0

        for step_index in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            # 测试阶段只执行动作和统计结果，不调用 agent.learn()。
            total_reward += reward
            state = next_state
            steps_used = step_index + 1

            if done:
                reached_goal = True
                test_success_count += 1
                break

        test_reward_sum += total_reward
        test_steps_sum += steps_used
        test_collisions_sum += env.hit_wall_num

        episode_number = episode_index + 1
        print(
            f"测试第 {episode_number} 局，"
            f"总奖励：{total_reward}，"
            f"到达终点：{reached_goal}，"
            f"步数：{steps_used}，"
            f"撞墙次数：{env.hit_wall_num}"
        )

    test_success_rate = test_success_count / test_episode_count * 100
    average_reward = test_reward_sum / test_episode_count
    average_steps = test_steps_sum / test_episode_count
    average_collisions = test_collisions_sum / test_episode_count

    print("\n测试汇总（100 局）")
    print(f"成功率：{test_success_rate:.1f}%")
    print(f"平均总奖励：{average_reward:.2f}")
    print(f"平均步数（包含超时局）：{average_steps:.2f}")
    print(f"平均撞墙次数：{average_collisions:.2f}")


if __name__ == "__main__":
    main()
