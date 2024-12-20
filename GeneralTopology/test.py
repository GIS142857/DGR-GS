import heapq
import pandas as pd


def distance(pos1, pos2):
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def is_acute_triangle(a, b, c):
    # 将三边长排序
    sides = sorted([a, b, c])
    # 使用不等式判断是否为锐角三角形
    return sides[0] ** 2 + sides[1] ** 2 >= sides[2] ** 2


# ---------------------- step 1 ---------------------- #
# df = pd.read_excel("HSL_tree2000.xlsx")
#
# print(df.head())
# df['10m'] = [[] for _ in range(len(df))]
# for i in df.index:
#     print(i)
#     for j in df.index:
#         if i == j:
#             continue
#         dist = distance([df.iloc[i]["gx"], df.iloc[i]["gy"]], [df.iloc[j]["gx"], df.iloc[j]["gy"]])
#         if dist <= 10:
#             df.iloc[i]['10m'].append(df.iloc[j]["FID"])
#
#     # print(df.iloc[i])
# print(df.head())
# df.to_csv("HSL_tree_10m.csv", index=False)


# ---------------------- step 2 ---------------------- #
df = pd.read_csv("HSL_tree_10m.csv")
print(df.head(5))
df['two_points'] = [[] for _ in range(len(df))]
df['sample_area'] = ["-" for _ in range(len(df))]
for i in df.index:
    pos0 = [df.loc[i, "gx"], df.loc[i, "gy"]]
    neighbors = df.loc[i, "10m"]
    FID0 = df.loc[i, "FID"]
    if neighbors == '[]':
        continue
    neighbors = neighbors[1:-1].split(",")
    if len(neighbors) < 2:
        continue
    # print(pos1, neighbors)
    dist_dict = {}
    for fid in neighbors:
        posi = [df[df["FID"] == int(fid)]["gx"].values[0], df[df["FID"] == int(fid)]["gy"].values[0]]
        dist_dict[int(fid)] = distance(pos0, posi)
    smallest_keys = heapq.nsmallest(2, dist_dict, key=dist_dict.get)
    df.iloc[i]["two_points"].append(smallest_keys)
    FID1 = smallest_keys[0]
    pos1 = [df[df["FID"] == FID1]["gx"].values[0], df[df["FID"] == FID1]["gy"].values[0]]
    FID2 = smallest_keys[1]
    pos2 = [df[df["FID"] == FID2]["gx"].values[0], df[df["FID"] == FID2]["gy"].values[0]]
    a, b, c = dist_dict[FID1], dist_dict[FID2], distance(pos1, pos2)
    print(a, b, c)
    if is_acute_triangle(a, b, c):
        tag0, tag1, tag2 = df[df["FID"] == FID0]["tag"].values[0], df[df["FID"] == FID1]["tag"].values[0], \
        df[df["FID"] == FID2]["tag"].values[0]
        spec0, spec1, spec2 = df[df["FID"] == FID0]["species"].values[0], df[df["FID"] == FID1]["species"].values[0], \
        df[df["FID"] == FID2]["species"].values[0]

        df.loc[i, "area_tag"] = str(tag0) + "-" + str(tag1) + "-" + str(tag2)
        df.loc[i, "area_spec"] = str(spec0) + "-" + str(spec1) + "-" + str(spec2)
        neighbors1 = df[df["FID"] == FID1]["10m"].values[0]
        neighbors2 = df[df["FID"] == FID2]["10m"].values[0]
        neighbors1 = neighbors1[1:-1].split(",")
        neighbors2 = neighbors2[1:-1].split(",")
        pos1_island, pos2_island = True, True

        for fid1_i in neighbors1:
            if fid1_i in [FID0, FID2]:
                continue
            posA = [df[df["FID"] == int(fid1_i)]["gx"].values[0], df[df["FID"] == int(fid1_i)]["gy"].values[0]]
            dist1_i = distance(pos1, posA)
            if dist1_i < min(a, c):
                pos1_island = False

        for fid2_i in neighbors2:
            if fid2_i in [FID0, FID1]:
                continue
            posB = [df[df["FID"] == int(fid2_i)]["gx"].values[0], df[df["FID"] == int(fid2_i)]["gy"].values[0]]
            dist2_i = distance(pos2, posB)
            if dist2_i < min(b, c):
                pos2_island = False
        df.loc[i, "sample_area"] = "select"
        if pos1_island and pos2_island:
            df.loc[i, "sample_area"] = "select_final"
print(df.head(10))
df.to_csv("HSL_tree_sample_area.csv", encoding="gbk")

# ---------------------- step 3 ---------------------- #
df = pd.read_csv("HSL_tree_10m.csv")
