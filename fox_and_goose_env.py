import gym
import numpy as np
from gym import spaces
from copy import deepcopy

# 定义动作空间的枚举，涵盖了狐狸和鹅在棋盘上可能的移动方式
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_DIAGONAL_UP_LEFT = 4
ACTION_DIAGONAL_UP_RIGHT = 5
ACTION_DIAGONAL_DOWN_LEFT = 6
ACTION_DIAGONAL_DOWN_RIGHT = 7

class FoxAndGooseEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FoxAndGooseEnv, self).__init__()
        # 初始化棋盘，用数字替换对应字符来表示不同元素，替换为新的初始棋盘格式
        self.board = [
            [' ', ' ', '.', '.', '.', ' ', ' '],
            [' ', ' ', '.', '.', '.', ' ', ' '],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['G', '.', '.', 'F', '.', '.', 'G'],
            ['G', 'G', 'G', 'G', 'G', 'G', 'G'],
            [' ', ' ', 'G', 'G', 'G', ' ', ' '],
            [' ', ' ', 'G', 'G', 'G', ' ', ' ']
        ]
        self.fox_illegal = 0
        self.goose_illegal = 0
        self.rounds_played = 0
        self.winner = ''
        self.visited_positions_fox = set()  # 记录狐狸已访问过的位置
        self.visited_positions_goose = set()  # 记录鹅已访问过的位置

        # 定义动作空间，离散的动作，这里有8种可能动作（根据定义的枚举）
        self.action_space = spaces.Discrete(8)
        # 定义状态空间，简单将棋盘展平为一维向量作为状态表示（可根据实际优化）
        self.observation_space = spaces.Box(low=0, high=3, shape=(49,), dtype=np.int8)

    # def step(self, action):
    #     """
    #     根据给定动作执行一步游戏，更新环境状态，并返回相关信息，包括新状态、奖励、是否结束以及其他额外信息。
    #     """
    #     # 确定当前是狐狸还是鹅行动（假设先鹅后狐狸交替进行回合）
    #     is_fox = self.rounds_played % 2 == 1

    #     # 根据当前行动角色获取起始位置（找到狐狸或鹅在棋盘上的位置）
    #     if is_fox:
    #         start_pos = self._find_fox_position()
    #     else:
    #         start_pos = self._find_goose_position()

    #     # 根据动作计算目标位置（依据具体动作对应的移动规则）
    #     target_pos = self._calculate_target_position(start_pos, action)

    #     moves = [start_pos, target_pos]

    #     # 判断移动是否合法，更新棋盘等（详细的合法性判断及棋盘更新逻辑）
    #     valid_move = self._is_valid_move(is_fox, moves)
    #     if valid_move:
    #         self._update_board(is_fox, moves)
    #         self.rounds_played += 1
    #         done = self._check_win()
    #         reward = self._calculate_reward(is_fox, done)
    #         # 增加探索新位置奖励（狐狸或鹅到达新位置时）
    #         if is_fox:
    #             pos_tuple = tuple(target_pos)
    #             if pos_tuple not in self.visited_positions_fox:
    #                 self.visited_positions_fox.add(pos_tuple)
    #                 reward += 0.1
    #         else:
    #             pos_tuple = tuple(target_pos)
    #             if pos_tuple not in self.visited_positions_goose:
    #                 self.visited_positions_goose.add(pos_tuple)
    #                 reward += 0.1
    #     else:
    #         self._increase_illegal_moves(is_fox)
    #         reward = -1  # 非法移动给予负奖励
    #         done = False

    #     # 返回新的状态、奖励、是否结束以及空字典（用于存放可能的额外信息，目前为空）
    #     return self._get_observation(), reward, done, {}
    def step(self, action):
        """
        根据给定动作执行一步游戏，更新环境状态，并返回相关信息，包括新状态、奖励、是否结束以及其他额外信息。
        """
        # 确定当前是狐狸还是鹅行动（假设先鹅后狐狸交替进行回合）
        is_fox = self.rounds_played % 2 == 1

        # 获取当前状态
        start_pos = self._find_fox_position() if is_fox else self._find_goose_position()

        # 根据动作计算目标位置
        target_pos = self._calculate_target_position(start_pos, action,is_fox)
        moves = [start_pos, target_pos]

        # 判断移动是否合法，更新棋盘等（详细的合法性判断及棋盘更新逻辑）
        valid_move = self._is_valid_move(is_fox, moves)
        if valid_move:
            self._update_board(is_fox, moves)  # 更新棋盘
            self.rounds_played += 1
            
            # 判断是否捕获鹅
            capture = self._is_valid_move(is_fox, moves, capture=True)
            
            # 获取当前鹅的数量
            goose_count_before = len(self._get_goose_positions())
            
            # 计算游戏是否结束
            done = self._check_win()

            # 计算奖励，传递是否捕获的标志
            reward = self._calculate_reward(is_fox, done, capture, goose_count_before)

            # 返回新的状态、奖励、是否结束以及空字典（用于存放可能的额外信息，目前为空）
            return self._get_observation(), reward, done, {}

        else:
            # 如果移动不合法，增加非法移动的次数并给予负奖励
            self._increase_illegal_moves(is_fox)
            reward = -1  # 非法移动给予负奖励
            done = False

        # 返回新的状态、奖励、是否结束以及空字典（用于存放可能的额外信息，目前为空）
        return self._get_observation(), reward, done, {}




    def reset(self):
        """
        重置环境到初始状态，重新初始化棋盘及相关游戏状态变量。
        """
        self.board = [
            [' ', ' ', '.', '.', '.', ' ', ' '],
            [' ', ' ', '.', '.', '.', ' ', ' '],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['G', '.', '.', 'F', '.', '.', 'G'],
            ['G', 'G', 'G', 'G', 'G', 'G', 'G'],
            [' ', ' ', 'G', 'G', 'G', ' ', ' '],
            [' ', ' ', 'G', 'G', 'G', ' ', ' ']
        ]
        self.fox_illegal = 0
        self.goose_illegal = 0
        self.rounds_played = 0
        self.winner = ''
        self.visited_positions_fox = set()
        self.visited_positions_goose = set()
        return self._get_observation()

    def render(self, mode='human'):
        """
        可视化展示当前棋盘状态，方便直观查看游戏进展。
        """
        print('  0 1 2 3 4 5 6')
        for i in range(len(self.board)):
            txt = str(i) + ' '
            for j in range(len(self.board[i])):
                element = self.board[i][j]
                if element == ' ':
                    txt += ' ' + " "
                elif element == 'G':
                    txt += 'G' + " "
                elif element == '.':
                    txt += '.' + " "
                elif element == 'F':
                    txt += 'F' + " "
            print(txt)
        print('')

    def _find_fox_position(self):
        """
        找到狐狸在棋盘上的位置，遍历棋盘寻找字符'F'所在坐标。
        """
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 'F':
                    return [i, j]

    def _find_goose_position(self):
        """
        找到所有鹅在棋盘上的位置，遍历棋盘寻找字符'G'所在坐标。
        """
        positions = []
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 'G':
                    positions.append([i, j])
        return positions

    # def _calculate_target_position(self, start_pos, action):
    #     """
    #     根据起始位置和动作准确计算目标位置，考虑棋盘边界情况，避免越界。
    #     """
    #     row, col = start_pos
    #     if action == ACTION_UP:
    #         row = max(row - 1, 0)
    #     elif action == ACTION_DOWN:
    #         row = min(row + 1, len(self.board) - 1)
    #     elif action == ACTION_LEFT:
    #         col = max(col - 1, 0)
    #     elif action == ACTION_RIGHT:
    #         col = min(col + 1, len(self.board[0]) - 1)
    #     elif action == ACTION_DIAGONAL_UP_LEFT:
    #         row = max(row - 1, 0)
    #         col = max(col - 1, 0)
    #     elif action == ACTION_DIAGONAL_UP_RIGHT:
    #         row = max(row - 1, 0)
    #         col = min(col + 1, len(self.board[0]) - 1)
    #     elif action == ACTION_DIAGONAL_DOWN_LEFT:
    #         row = min(row + 1, len(self.board) - 1)
    #         col = max(col - 1, 0)
    #     elif action == ACTION_DIAGONAL_DOWN_RIGHT:
    #         row = min(row + 1, len(self.board) - 1)
    #         col = min(col + 1, len(self.board[0]) - 1)
    #     return [row, col]
    def _calculate_target_position(self, start_pos, action, is_fox):
        """
        根据起始位置和动作准确计算目标位置，考虑棋盘边界情况，避免越界。
        如果是狐狸，则处理狐狸的移动逻辑；如果是鹅，则处理鹅的移动逻辑。
        """
        row, col = start_pos

        if is_fox:
            # 狐狸的移动规则
            if action == ACTION_UP:
                row = max(row - 1, 0)
            elif action == ACTION_DOWN:
                row = min(row + 1, len(self.board) - 1)
            elif action == ACTION_LEFT:
                col = max(col - 1, 0)
            elif action == ACTION_RIGHT:
                col = min(col + 1, len(self.board[0]) - 1)
            elif action == ACTION_DIAGONAL_UP_LEFT:
                row = max(row - 1, 0)
                col = max(col - 1, 0)
            elif action == ACTION_DIAGONAL_UP_RIGHT:
                row = max(row - 1, 0)
                col = min(col + 1, len(self.board[0]) - 1)
            elif action == ACTION_DIAGONAL_DOWN_LEFT:
                row = min(row + 1, len(self.board) - 1)
                col = max(col - 1, 0)
            elif action == ACTION_DIAGONAL_DOWN_RIGHT:
                row = min(row + 1, len(self.board) - 1)
                col = min(col + 1, len(self.board[0]) - 1)
        else:
            # 鹅的移动规则
            if action == ACTION_UP:
                row = max(row - 1, 0)
            elif action == ACTION_DOWN:
                row = min(row + 1, len(self.board) - 1)
            elif action == ACTION_LEFT:
                col = max(col - 1, 0)
            elif action == ACTION_RIGHT:
                col = min(col + 1, len(self.board[0]) - 1)
            elif action == ACTION_DIAGONAL_UP_LEFT:
                row = max(row - 1, 0)
                col = max(col - 1, 0)
            elif action == ACTION_DIAGONAL_UP_RIGHT:
                row = max(row - 1, 0)
                col = min(col + 1, len(self.board[0]) - 1)
            elif action == ACTION_DIAGONAL_DOWN_LEFT:
                row = min(row + 1, len(self.board) - 1)
                col = max(col - 1, 0)
            elif action == ACTION_DIAGONAL_DOWN_RIGHT:
                row = min(row + 1, len(self.board) - 1)
                col = min(col + 1, len(self.board[0]) - 1)

        return [row, col]

    # def _is_valid_move(self, is_fox, moves):
    #     """
    #     判断移动是否合法，综合考虑多方面规则，如起始位置是否正确、目标位置是否可达、是否符合角色移动规则等。
    #     """
    #     from_pos, to_pos = moves
    #     from_row, from_col = from_pos
    #     to_row, to_col = to_pos

    #     # 测试起始位置是否合理
    #     if from_row < 0 or from_row >= len(self.board) or from_col < 0 or from_col >= len(self.board[0]):
    #         return False
    #     if (not is_fox and self.board[from_row][from_col]!= 'G') or (is_fox and self.board[from_row][from_col]!= 'F'):
    #         return False

    #     # 测试目标位置是否在棋盘内且为空（可移动到的位置）
    #     if to_row < 0 or to_row >= len(self.board) or to_col < 0 or to_col >= len(self.board[0]) or self.board[to_row][to_col]!= '.':
    #         return False

    #     # 检查移动距离是否合规（这里限制单步移动最多一格或特定条件下的多格捕获移动）
    #     row_diff = abs(from_row - to_row)
    #     col_diff = abs(from_col - to_col)
    #     if row_diff > 2 or col_diff > 2:
    #         return False

    #     # 对于鹅，正常只能单步移动（相邻格），除非是特殊的连续捕获情况（暂未详细实现连续捕获逻辑）
    #     if not is_fox and (row_diff > 1 or col_diff > 1):
    #         return False

    #     # 对于狐狸，单步移动或符合捕获规则的多步移动（跨越一只鹅）才合法
    #     if is_fox:
    #         if row_diff > 1 or col_diff > 1:
    #             mid_row = (from_row + to_row) // 2
    #             mid_col = (from_col + to_col) // 2
    #             if self.board[mid_row][mid_col]!= 'G':
    #                 return False
    #     return True
    def _is_valid_move(self, is_fox, moves, capture=False):
        """
        判断移动是否合法，综合考虑多方面规则，如起始位置是否正确、目标位置是否可达、是否符合角色移动规则等。
        """
        from_pos, to_pos = moves
        from_row, from_col = from_pos
        to_row, to_col = to_pos

        # 测试起始位置是否合理
        if from_row < 0 or from_row >= len(self.board) or from_col < 0 or from_col >= len(self.board[0]):
            return False
        if (not is_fox and self.board[from_row][from_col] != 'G') or (is_fox and self.board[from_row][from_col] != 'F'):
            return False

        # 测试目标位置是否在棋盘内且为空（可移动到的位置）
        if to_row < 0 or to_row >= len(self.board) or to_col < 0 or to_col >= len(self.board[0]):
            return False

        # 对于非捕获移动，目标位置必须为空
        if not capture and self.board[to_row][to_col] != '.':
            return False

        # 检查移动距离是否合规（这里限制单步移动最多一格或特定条件下的多格捕获移动）
        row_diff = abs(from_row - to_row)
        col_diff = abs(from_col - to_col)
        if row_diff > 2 or col_diff > 2:
            return False

        # 对于捕获移动，检查中间位置是否有鹅
        if capture:
            mid_row = (from_row + to_row) // 2
            mid_col = (from_col + to_col) // 2
            if self.board[mid_row][mid_col] != 'G' or self.board[to_row][to_col] != '.':
                return False

        return True
    
    def _update_board(self,is_fox, moves, capture=False):
        """
        根据合法移动更新棋盘状态，移动角色位置并处理可能的捕获情况。
        """
        from_pos, to_pos = moves
        from_row, from_col = from_pos
        to_row, to_col = to_pos

        # 更新棋盘状态
        self.board[to_row][to_col] = self.board[from_row][from_col]
        self.board[from_row][from_col] = '.'

        # 如果是捕获移动，移除中间的鹅
        if capture:
            mid_row = (from_row + to_row) // 2
            mid_col = (from_col + to_col) // 2
            self.board[mid_row][mid_col] = '.'

    def _check_win(self):
        """
        检查游戏是否有一方获胜，分别判断狐狸是否困住（无法移动）以及鹅的数量是否过少。
        """
        goose_count = 0
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 'G':
                    goose_count += 1

        # 如果鹅的数量小于4只，狐狸获胜
        if goose_count < 4:
            self.winner = 'F'
            return True

        fox_pos = self._find_fox_position()
        fox_row, fox_col = fox_pos

        # 检查狐狸是否能移动（一步移动或者可以捕获鹅的情况）
        for i in range(max(fox_row - 1, 0), min(fox_row + 2, len(self.board))):
            for j in range(max(fox_col - 1, 0), min(fox_col + 2, len(self.board[0]))):
                if (i == fox_row or j == fox_col) or ((fox_row % 2) == (fox_col % 2)):
                    if self.board[i][j] == '.':
                        return False  # 狐狸能移动一步
                    if self.board[i][j] == 'G':
                        capture_pos = [i + i - fox_row, j + j - fox_col]
                        if capture_pos[0] >= 0 and capture_pos[0] < len(self.board) and capture_pos[1] >= 0 and capture_pos[1] < len(self.board[0]):
                            if self.board[capture_pos[0]][capture_pos[1]] == '.':
                                return False  # 狐狸能捕获一只鹅

        # 狐狸无法移动，鹅获胜
        self.winner = 'G'
        return True

    def _calculate_reward(self, is_fox, done, capture=False, goose_count_before=None):
        """
        根据行动角色、游戏是否结束以及是否捕获鹅来计算奖励。
        """
        reward = 0

        # 如果游戏已经结束，根据赢家和角色计算奖励
        if done:
            if is_fox:
                return 1 if self.winner == 'F' else -1  # 狐狸胜利 +1，失败 -1
            else:
                return 1 if self.winner == 'G' else -1  # 鹅胜利 +1，失败 -1

        # 如果捕获了鹅，给予额外奖励
        if capture:
            reward += 5  # 捕获鹅奖励

        # 获取当前鹅的数量
        goose_positions = self._get_goose_positions()  # 获取当前所有鹅的位置
        goose_count_after = len(goose_positions)  # 计算当前鹅的数量

        # 如果鹅数量减少，狐狸获得奖励
        if is_fox and goose_count_after < goose_count_before:
            goose_weight = 1  # 可调节的权重
            reward += goose_weight * (goose_count_before - goose_count_after)  # 根据鹅减少的数量调整奖励

        # 返回奖励值
        return reward


    def _increase_illegal_moves(self, is_fox):
        """
        增加对应角色的非法移动次数记录。
        """
        if is_fox:
            self.fox_illegal += 1
        else:
            self.goose_illegal += 1

    def _get_observation(self):
        """
        获取当前环境的状态表示，直接将棋盘展平为一维向量，元素类型为np.int8符合要求。
        """
        # 需要将棋盘元素转换为数字表示后再展平，这里简单定义一个映射
        element_mapping = {
            ' ': 0,
            '.': 1,
            'G': 2,
            'F': 3
        }
        num_board = [[element_mapping[ele] for ele in row] for row in self.board]
        return np.array(num_board).flatten().astype(np.int8)