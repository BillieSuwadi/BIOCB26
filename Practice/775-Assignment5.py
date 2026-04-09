#!/usr/bin/env python3
import sys
#Problem 15-2
def palidrome(s: str) -> str:
    if not s:
        return ""

    n = len(s)
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = 1

    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])

    for k in range(n):
        print(dp[k])

    left = []
    right = []
    i, j = 0, n - 1

    while i <= j:
        if i == j:
            left.append(s[i])
            break
        if s[i] == s[j]:
            left.append(s[i])
            right.append(s[j])
            i += 1
            j -= 1
        elif dp[i + 1][j] >= dp[i][j - 1]:
            i += 1
        else:
            j -= 1

    return "".join(left + right[::-1])


def Printing_Neatly(l, n, M):
    """
    :param l: 包含每个单词的长度的数组
    :param n: 单词的个数
    :param M:  一行所能容纳的字符个数
    :return:
    """
    Ext = [[0 for i in range(n + 1)] for j in range(n + 1)]
    LCost = [[0 for i in range(n + 1)] for j in range(n + 1)]
    for i in range(1, n + 1):
        Ext[i][i] = M - l[i - 1]
        for j in range(i + 1, n + 1):
            Ext[i][j] = Ext[i][j - 1] - l[j - 1] - 1  # compute extra space
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            if Ext[i][j] < 0:
                LCost[i][j] = sys.maxsize  # this line don’t fit, to set the cost is infinity
            elif Ext[i][j] >= 0 and j == n:
                LCost[i][j] = 0
            else:
                LCost[i][j] = (Ext[i][j]) ^ 3
    C = [0 for i in range(n + 1)]
    # 计算最小代价而且找到最小代价的每行换行处。
    # C[j]意味着从单词1到单词j的最优总代价(代价=立方和)

    # R[]用于打印解决方式
    R = [0 for i in range(n + 1)]
    for j in range(1, n + 1):
        C[j] = sys.maxsize
        for i in range(1, j + 1):
            if C[i - 1] != sys.maxsize and (LCost[i][j] + C[i - 1]) < C[j]:
                C[j] = LCost[i][j] + C[i - 1]
                R[j] = i

    print_lines(R, n)
    return


def print_lines(R, j):
    no = 0
    i = R[j]
    if i == 1:
        no = 1
    else:
        no = print_lines(R, i - 1) + 1
    print("Line number:", no, end=", ")
    for t in range(i, j):
        print("", word[t - 1], end=" ")
    print("")
    return no  # 行号


if __name__ == "__main__":
    # Problem 15-2
    #print(palidrome('character'))

    word = ["optimal", "substructure", "in", "the", "following", "way"]
    n = len(word)
    l = [0 for i in range(n)]
    for i in range(n):
        l[i] = len(word[i])
    Printing_Neatly(l, n, 20)