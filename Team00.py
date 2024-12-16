# Fox and Geese manual play example
# JC4004 Computational Intelligence 2024-25
import torch
from ppo_agent import PPOAgent
import numpy as np

class Player:
    def __init__(self):
        state_size = 49  # 根据棋盘状态的实际维度来确定，这里假设棋盘是7x7，展平后为49维
        action_size = 8  # 假设动作空间有8种可能的动作（前面定义的8个方向等情况）
        self.fox_agent = PPOAgent(state_size, action_size)
        self.goose_agent = PPOAgent(state_size, action_size)
        self.fox_agent.load_model('saved_models/fox_agent_episode_900.pt')  # 加载狐狸模型权重，需替换为实际路径
        self.goose_agent.load_model('saved_models/goose_agent_episode_900.pt')  # 加载鹅模型权重，需替换为实际路径
        self.role = None 
  # =================================================
  # Print the board
    def print_board(self,board):

      # Prints the current board
      print('')
      print('  0 1 2 3 4 5 6')
      for i in range(len(board)):
        txt = str(i) + ' '
        for j in range(len(board[i])):
          txt += board[i][j] + " "
        print(txt)
      print('')

# =================================================



    def play_fox(self, board):
            # First, print the current board
        self.print_board(board)
        """作为狐狸进行游戏时调用的方法，利用加载的狐狸PPO模型生成动作。"""
        self.role = 'F'
        state = self._preprocess_board(board)
        action, _ = self.fox_agent.act(state)
        move = self._action_to_move(state, action)
        return move

    def play_goose(self, board):
                    # First, print the current board
        self.print_board(board)
        """作为鹅进行游戏时调用的方法，利用加载的鹅PPO模型生成动作。"""
        self.role = 'G'
        state = self._preprocess_board(board)
        action, _ = self.goose_agent.act(state)
        move = self._action_to_move(state, action)
        return move

    def _preprocess_board(self, board):
        """将二维棋盘列表转换为适合 PPO 模型输入的一维张量。"""
        mapping = {'F': 3, 'G': 2, ' ': 0, '.': 1}  # 定义字符到数值的映射字典
        encoded_board = [[mapping[element] for element in row] for row in board]  # 对棋盘元素进行编码转换
        flattened_board = np.array(encoded_board).flatten()
        tensor_board = torch.from_numpy(flattened_board).float().unsqueeze(0)
        return tensor_board

    def _action_to_move(self, state, action):
        """根据动作编号将其转换为具体的移动坐标变化，并返回格式符合要求的移动路径列表。"""
        if isinstance(state, torch.Tensor):
            state = state.squeeze(0).detach().cpu().numpy()  # 去除batch维度并转换为numpy数组
        state = state.reshape(7, 7)  # 将一维数组重塑为二维数组

        row, col = self._find_piece_position(state, self.role)

        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # 左上、上、右上
            (0, -1), (0, 1),             # 左、右
            (1, -1), (1, 0), (1, 1)      # 左下、下、右下
        ]
        direction = directions[action]
        new_row = row + direction[0]
        new_col = col + direction[1]

        move = [[row, col], [new_row, new_col]]

        if self.role == 'F':
            # 狐狸的捕获移动处理，如果可以捕获鹅，则添加捕获路径
            mid_row = new_row
            mid_col = new_col
            target_row = mid_row + direction[0]
            target_col = mid_col + direction[1]

            if self._is_valid_position(state, mid_row, mid_col) and state[mid_row][mid_col] == 2:  # 2表示鹅
                if self._is_valid_position(state, target_row, target_col) and state[target_row][target_col] == 1:  # 1表示空位
                    move.append([target_row, target_col])

        return move

    def _find_piece_position(self, state, piece):
        """在给定状态中找到指定棋子（狐狸或鹅）的位置。"""
        piece_code = 3 if piece == 'F' else 2  # 狐狸编码为3，鹅编码为2
        for row in range(len(state)):
            for col in range(len(state[row])):
                if state[row][col] == piece_code:
                    return row, col
        return None

    def _is_valid_position(self, state, row, col):
        """检查给定的行和列位置是否在棋盘内且是有效可移动的位置（非游戏区域外的空格）。"""
        return 0 <= row < len(state) and 0 <= col < len(state[row]) and state[row][col] != 0  # 0表示不可移动位置


#   # Play one move as a fox
#   def play_fox(self, board):

#     # First, print the current board
#     self.print_board(board)

#     # Find where the fox is
#     fox_pos = [0,0]
#     for i in range(len(board)):
#       for j in range(len(board[i])):
#         if board[i][j] == 'F':
#           fox_pos = [i,j]
#           break

#     move = [fox_pos]

#     # Get player input for new fox position
#     cont = True
#     print("Fox plays next!")
#     print("Fox is in position (" + str(fox_pos[0]) + "," + str(fox_pos[1]) + ")")
#     while cont:      
#       new_pos = [int(i) for i in input("Give the new position for the fox (row,column): ").split(',')]
#       move.append(new_pos)
#       if abs(move[-1][0]-move[-2][0]) > 1 or abs(move[-1][1]-move[-2][1]) > 1:
#         inp = input("Do you want to make another move [y/N]? ")
#         if inp != 'y' and inp != 'Y':
#           cont = False
#       else:
#         cont = False

#     print(move)

#     return move

#   # =================================================
#   # Play one move as a goose
#   def play_goose(self, board):

#     # First, print the current board
#     self.print_board(board)

#     # Get goose start position
#     print("Goose plays next!")
#     goose_pos = [int(i) for i in input("Give the position of the goose to move (row,column): ").split(',')]

#     move = [goose_pos]

#     # Get player input for goose next position
#     new_pos = [int(i) for i in input("Give the new position (row,column): ").split(',')]
#     move.append(new_pos)

#     print(move)

#     return move

# # ==== End of file