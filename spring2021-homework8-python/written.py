def check(blah):
    for x in blah:
        for y in blah:
            if x[0] in y and x[1] in y and x[2] in y:
                blah.remove(y)
    return blah

def poo(seq):
    blah = []
    for i in seq:
        for j in seq:
            for k in seq:
                for b in seq:
                    if (i + j + k + b) % 2 == 0:
                    # if check((i, j, k), blah):
                        blah.append((i, j, k, b))
    # for i in seq:
    #     for j in seq:
    #         if (i + j) % 2 == 0:
    #             if (i, j) not in blah and (j, i) not in blah:
    #                 blah.append((i, j))
    return blah

seqfs = [1, 2, 3, 4, 5]
dfsfs = poo(seqfs)

print(len(dfsfs))


            count1 = 
# def mystery(A, k):
#     n = len(A)
#     print("n: " + str(n))
#     print("k: " + str(k))
#     print("")
#     if k == 0:
#         print("EMPTY SET")
#         return []
#     if k < 0 or n == 0:
#         #print("{-1}")
#         return [-1]
#     # print("")
#     B = mystery(A[:-1], k)
#     # print(B)
#     # print("")
#     if B != [-1]:
#         # print("FIRST B")
#         return B
#     B = mystery(A[:-1], k - A[-1])
#     print(B)
#     if B != [-1]:
#         # print("UNION B")
#         # print(A[n-1])
#         return B + A[n-1]
#     return {-1}

# A = [3, 1, 7, 12, 8]
# print(mystery(A, 16))
# def austin(principle: float, payments: float, interest: float, count: int):
#     if count == 0:
#         return 
# import math
# def austin(principle: float, payments: float, interest: float):
#     """
#     Inputs: 
#         - principle
#         - payments
#         - interest
#     """
#     count = 0
#     p_cop = principle
#     print(type(p_cop))
    
#     while p_cop > 0:
#         p_cop = p_cop - (payments - p_cop*interest)
#         print(p_cop)
#         count += 1
        
#     # for i in range(principle, 0.0, -1):
#     #     if p_comp != 0:
#     #         p_cop = p_cop - (payments - p_cop*interest)
#     #         print(type(p_cop))
#     #         count += 1
#     return count
    
# print(austin(5471.0, 569.0, 0.07))

# A = True
# B = True
# C = True
# left = (not A) and (B ^ C)
# right = not ((not B) and A)
# print(not (left or right))
# x = 9
# print(x)
# if x < 10:
#     x = x + 10
#     print(x)
# if x > 10:
#     x = x - 10
#     print(x)
# else:
#     x = 10
#     print(x)
# print(x)
# A = True
# B = False
# left = A and (not B)
# right = A or (not B)
# print(left and right)
# print(A and B)


# import numpy as np
# import random
# import copy
# grapher = {1: [2, 3, 4, 5], 2: [1, 3, 5], 3: [1, 2, 4], 4: [1, 3, 5], 5: [1, 4, 5]}
# def scam(prev_scam):
#     scams = []
#     for item in prev_scam:
#         for value in grapher[item[-1]]:
#             path = item + [value]
#             scams.append(path)
#     return scams

# def paths(length):
#     paths = [[1]]
#     for index in range(length):
#         paths = scam(paths)
#     return paths

# # lengths = []
# # for i in range(0, 15):
# #     print(i)
# #     lengths.append(len(paths(i)))
# # print(len(paths(15)))
# # print(lengths)
# lengths_thing = [1, 4, 12, 40, 128, 416, 1344, 4352, 14080, 45568, 147456, 477184, 1544192, 4997120, 16171008]

# def check(n):
#     sn = 8*(lengths_thing[n-2] + lengths_thing[n-3])
#     actual = lengths_thing[n]
#     return sn == actual

# # for i in range(3, 15):
# #     print(check(i))

# num3shit = [3, 8, 22, 60]
# def check3(n):
#     return 2*(num3shit[n-1] + num3shit[n-2])
    
# for i in range(4, 20):
#     num3shit.append(check3(i))

# # print(num3shit)

# # print(check3(2))
# def help(edges: dict, n: int, combos: list):
#     combo_copy = copy.deepcopy(combos)
#     n_combos = []
#     for pair in combos:
#         pair_copy = copy.deepcopy(pair)
#         for node, edge in edges.items():
#             for x in edge:
                
#                 # print(pair[1])
#                 if pair[len(pair)-1] == node:
#                     temp = copy.deepcopy(pair_copy)
#                     # print("x: " + str(x))
#                     # print("temp: " + str(temp))
#                     a = temp.append(x)
#                     n_combos.append(temp)
#     return n_combos


# def graph_combos(edges: dict, n: int):
#     # list of lists
#     combos = []
#     for node in edges[1]:
#         combos.append([1, node])
    
#     i = 2
#     while i < n:
#         combos = help(edges, n, combos)
#         i+=1


#     return combos

# # print(len(graph_combos(edges, 3)))

# def pixel_matrix(m: int, n: int):
#     """
#     Inputs:
#         - m: an int representing how many rows the matrix will have
#         - n: an int representing how many columns the matrix will have
#     Outputs:
#         - a numpy array
#     """
#     # for the number of elements in each row
#     # add array the number of rows (m)
#     n_array = []
#     for i in range(0, n):
#         n_array.append(0)
    
#     pix_matrix = []
#     for x in range(0, m):
#         pix_matrix.append(n_array)
    
#     return np.array(pix_matrix)




# def disruption_measure(m, n):

#     dis_mat = []

#     for row in range(0, m):
#         temp = []
#         for column in range(0, n):
#             temp.append(random.randint(1, 10))
#         dis_mat.append(temp)
    
#     np_dis_mat = np.array(dis_mat)
#     return np_dis_mat

# dis = disruption_measure(5, 5)
# pixel = pixel_matrix(5, 5)


# def middle_choice(mat, column, row):
#     cur_coord = (row, column)
#     a = mat[row][column]
#     b = mat[row + 1][column - 1]
#     c = mat[row + 1][column]
#     d = mat[row + 1][column + 1]
#     temp_next_val = min((a+b),(a+c),(a+d))
#     if temp_next_val == (a+b):
#         temp_next_coord = (row + 1, column - 1)
#     elif temp_next_val == (a+c):
#         temp_next_coord = (row + 1, column)
#     else:
#         temp_next_coord = (row + 1, column + 1)
#     return (cur_coord, temp_next_coord, temp_next_val)

# def edge_choice(mat, column, row):
#     if column == 0:
#         cur_coord = (row, column)
#         a = mat[row][column]
#         b = mat[row + 1][column]
#         c = mat[row + 1][column + 1]
#         temp_next_val = min((a+b),(a+c))
#         if temp_next_val == (a+b):
#             temp_next_coord = (row + 1, column)
#         else:
#             temp_next_coord = (row + 1, column + 1)
#         return (cur_coord, temp_next_coord, temp_next_val)
#     else:
#         cur_coord = (row, column)
#         a = mat[row][column]
#         b = mat[row + 1][column]
#         c = mat[row + 1][column - 1]
#         temp_next_val = min((a+b),(a+c))
#         if temp_next_val == (a+b):
#             temp_next_coord = (row + 1, column)
#         else:
#             temp_next_coord = (row + 1, column - 1)
#         return (cur_coord, temp_next_coord, temp_next_val)


# def lowest_seam(mat, pixel):
#     m = mat.shape[0]
#     n = mat.shape[1]
#     path = []
#     # choose first
#     possible_first = []
#     if n != 1:
        
#         for column in range(0, n):
#             # not an edge
#             if column != 0 and column != n-1:
#                 possible_first.append(middle_choice(mat, column, 0))
#             else:
#                 possible_first.append(edge_choice(mat, column, 0))
#     #print("")
#     temp_low = float('inf')
#     temp_low_info = ()
#     for tup in possible_first:
#         if tup[2] < temp_low:
#             temp_low = tup[2]
#             temp_low_info = tup
#     # print(temp_low)
#     # print(temp_low_info)
#     path.append(temp_low_info[0])
#     path.append(temp_low_info[1])
#     prev_column = temp_low_info[1][1]
    
#     for row in range(1, m-1):
#         if prev_column != 0 and prev_column != n-1:
#             choice = middle_choice(mat, prev_column, row)
#             prev_column = choice[1][1]
#             path.append(choice[1])
#         else:
#             choice = edge_choice(mat, prev_column, row)
#             prev_column = choice[1][1]
#             path.append(choice[1])

#     for element in path:
#         pixel[element[0]][element[1]] = 1
    
#     #print(pixel)
#     return path

# low = lowest_seam(dis, pixel)
# print(pixel)
# print(low)
# print(dis)
# print("(current coordinate, next coordinate, value)")
# for pos in low:
#     print(pos)
                


# # def possible_seam(mat):
# #     """
# #     Inputs:
# #         - mat: a numpy matrix
# #     Outputs:
# #         - prints each possible combo
# #         - returns number of combos
# #     """
# #     # array of all combos
# #     all_mats = []
# #     m = mat.shape[0]
# #     n = mat.shape[1]
    
# #     cur_row = 0
# #     #while cur_row != m-1:
# #     for column in range(0, n):
# #         print("column: " + str(column))
# #         temp_combo = mat.copy()
# #         for row in range(0, m):
# #             print("row: " + str(row))
# #             # origin
# #             if row == 0:
# #                 temp_combo[row][column] = 1
# #                 prev_column = column
# #                 prev_row = row
# #                 break
# #                 #all_mats.append(temp_combo)
# #             else:
# #                 if column != 0 and column != n-1:
# #                     temp_combo[prev_row + 1][prev_column - 1] = 1



            
# #     # for row in range(0, m):
# #     #     print(str(row) + ": " + str(mat[row]))
# #     #     for column in range(0, n):
# #     #         mat[row][column] = column
# #     #iterate over columns
# #     # for row 
# #     return all_mats

# # seams = possible_seam(x)
# # for i in seams:
# #     print(i)
