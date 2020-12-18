# # 题目1 词典中最长的单词
# def dictionary(l):
#     import functools
#     res = []
#
#     def sort(x1, x2):
#         if len(x1) > len(x2):
#             return 1
#         elif len(x1) < len(x2):
#             return -1
#         else:
#             if x1 > x2:
#                 return -1
#             elif x1 < x2:
#                 return 1
#             else:
#                 return 0
#
#     l = sorted(l, key=functools.cmp_to_key(sort))
#
#     for i in range(len(l)):
#         if len(l[i]) == 1:
#             res.append(l[i])
#         elif l[i][:-1] in res:
#             res.append(l[i])
#     if res:
#         return res[-1]
#     else:
#         return ''
#     return res
#
#
# print(dictionary(["good", "go", "goo", 'to', 'too', 'tool']))
#
#
# # 题目2 无重复字符的最长子串
# def removeDuplicate(s):
#     s = list(s)
#     res = 0
#
#     if len(set(s)) == len(s):
#         return max(res, len(s))
#     else:
#         a = max(removeDuplicate(s[1:]), removeDuplicate(s[:-1]))
#         res = max(a, res)
#     return res
#
#
# print(removeDuplicate("abcdcba"))


# def heapify(freq, user, n, i):
#     largest = i
#     l = 2 * i + 1
#     r = 2 * i + 2
#
#     if l < n and freq[i] < freq[l]:
#         largest = l
#
#     if r < n and freq[largest] < freq[r]:
#         largest = r
#
#     if largest != i:
#         freq[i], freq[largest] = freq[largest], freq[i]
#         user[i], user[largest] = user[largest], user[i]
#
#         heapify(freq, user, n, largest)
#
#
# def heapSort(userMap, num):
#     userFreq = [0] * num
#     userId = ['0'] * num
#
#     for user in userMap:
#         freq = userMap[user]
#         if freq > userFreq[0] or (freq == userFreq[0] and user > user[0]):
#             userFreq[0] = freq
#             userId[0] = user
#             for i in range(num):
#                 heapify(userFreq, userId, num, i)
#
#             for i in range(num - 1, 0, -1):
#                 userFreq[i], userFreq[0] = userFreq[0], userFreq[i]
#                 userId[i], userId[0] = userId[0], userId[i]
#                 heapify(userFreq, userId, i, 0)
#
#     return userFreq, userId
#
#
# user_ID_Freq_map = {'100': 10,
#                     '101': 8,
#                     '102': 124,
#                     '103': 13,
#                     '104': 124}
#
# userF, userI = heapSort(user_ID_Freq_map, 3)
# print(userI)

#
# import os
# import pandas as pd
#
#
# # 首先读取系统最基本的玩家日志，日志包含两列，一列为进攻userId，一列为防守userId，默认进攻方玩家为在线状态
# # 默认系统文件存储在csv文件中，csv第一行为玩家类型标识符：[attacker, defender], userId 默认以string类型存储，范围1~1e8
# def separateUsers(user_root_dir, store_root_dir):
#
#     """
#     本函数实现遍历系统玩家活跃日志，将不同玩家按照userId分文件存储，同样存储进csv文件。
#     周活跃用户为1e7级别，则利用Id分段，每个文件存储1000名玩家活跃记录, 存储文件数为1e4级别。
#
#     :param store_root_dir: 系统存储玩家活跃日志根目录
#     :param user_root_dir: 分段数据存储根目录
#     """
#
#     # 获取根目录下所有玩家活跃日志文件名
#     file_names = os.listdir(user_root_dir)
#
#     for file_name in file_names:
#         df = pd.read_csv(os.path.join(user_root_dir, file_name))
#         for i in range(df.count()['attacker']):
#             userId = int(df.iloc[i]['attacker'])
#             # 利用userId计算该用户活跃记录应该存在哪个文件
#             temp = {'attacker': str(userId)}
#             temp = pd.DataFrame(temp)
#             stored_file_name = os.path.join(store_root_dir, str(userId // 100 + 1) + '.csv')
#             temp.to_csv(stored_file_name, mode='a', header=False)
#
#
# def heapify(freq_arr, user_arr):
#
#     root = freq_arr[0]
#     idx = 0
#     left = 2 * idx + 1
#     right = 2 * idx + 2
#     flag = True
#     while (root > freq_arr[left] or root > freq_arr[right]) and flag:
#         if freq_arr[left] <= freq_arr[right]:
#             freq_arr[idx], freq_arr[left] = freq_arr[left], freq_arr[idx]
#             user_arr[idx], user_arr[left] = user_arr[left], user_arr[idx]
#             idx = left
#         else:
#             freq_arr[idx], freq_arr[right] = freq_arr[right], freq_arr[idx]
#             user_arr[idx], user_arr[right] = user_arr[right], user_arr[idx]
#             idx = right
#         left = 2 * idx + 1
#         right = 2 * idx + 2
#
#         flag = left < len(freq_arr) and right < len(freq_arr)
#
#     return freq_arr, user_arr
#
#
# def getTopKusers(user_root_dir, store_root_dir, k):
#     """
#
#     :param user_root_dir: 系统日志根目录
#     :param store_root_dir: 用户分批存储根目录
#     :param k: 前K玩家
#     :return: 堆排序后前K玩家频率，前k玩家id
#     """
#
#     freq = [0] * k
#     users = [''] * k
#     separateUsers(user_root_dir, store_root_dir)
#
#     file_names = file_names = os.listdir(user_root_dir)
#
#     for file_name in file_names:
#         df = pd.read_csv(os.path.join(store_root_dir, file_name))
#         freq_map = df.apply(pd.value_counts)['attacker'].to_dict()
#         for userId in freq_map:
#             if freq_map[userId] > freq[0]:
#                 freq[0] = freq_map[userId]
#                 users[0] = userId
#                 heapify(freq, users)
#
#     return freq, users


# # 定义棋盘大小
# SIZE = 5
# # 定义一个全局变量，用于记录第total种巡游方式。
# total = 0
#
#
# def print_board(board):
#     for row in board:
#         for col in row:
#             print(str(col).center(4), end='')
#         print()
#
#
# def patrol(board, row, col, step=1):
#     if 0 <= row < SIZE and \
#             0 <= col < SIZE and \
#             board[row][col] == 0:
#         board[row][col] = step
#         # 当最后一步恰好等于 25（本案例5*5）时，打印输出巡游路线
#         if step == SIZE * SIZE:
#             global total
#             total += 1
#             print(f'第{total}种走法: ')
#             print_board(board)
#         # 下一步可能会走的位置
#         patrol(board, row - 2, col - 1, step + 1)
#         patrol(board, row - 1, col - 2, step + 1)
#         patrol(board, row + 1, col - 2, step + 1)
#         patrol(board, row + 2, col - 1, step + 1)
#         patrol(board, row + 2, col + 1, step + 1)
#         patrol(board, row + 1, col + 2, step + 1)
#         patrol(board, row - 1, col + 2, step + 1)
#         patrol(board, row - 2, col + 1, step + 1)
#         board[row][col] = 0
#
#
# def main():
#     # 生成5*5的棋盘
#     board = [[0] * SIZE for _ in range(SIZE)]
#     # 设定巡游起点为索引（4,4）
#     patrol(board, SIZE - 1, SIZE - 1)
#
#
# if __name__ == '__main__':
#     main()


# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
#
#
# class Solution:
#     def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
#         copy = ListNode(0)
#         p = copy
#         while True:
#             count = k
#             stack = []
#             tmp = head
#             while count and tmp:
#                 stack.append(tmp)
#                 tmp = tmp.next
#                 count -= 1
#             if count:
#                 p.next = head
#                 break
#             while stack:
#                 p.next = stack.pop()
#                 p = p.next
#             p.next = tmp
#             head = tmp
#
#         return copy.next
#
#     def test(self, head, k):
#         head = self.reverseKGroup(head, k)
#         while head is not None:
#             print(head.val)
#             head = head.next
#
#
# head = ListNode(5, None)
# head = ListNode(4, head)
# head = ListNode(3, head)
# head = ListNode(2, head)
# head = ListNode(1, head)
# s = Solution()
# s.test(head, 3)