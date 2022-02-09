import time
import sys
import json
from pyspark import SparkConf, SparkContext
from itertools import combinations
from operator import add

n_hashes = 75
bands = 75
rows = n_hashes // bands


if __name__ == "__main__":
    n_hashes = 75
    bands = 75
    rows = n_hashes // bands
    Tstart = time.time()
    conf = SparkConf().setAppName("Task-1").set("spark.executor.memory", "8g")
    sc = SparkContext(conf=conf)

    IF = sys.argv[1]
    reqd_JS = 0.05
    OF = sys.argv[2]
    RDD_ID = sc.textFile(IF)
    RDD_I= RDD_ID.map(json.loads)
    RDD_I =RDD_I.map(lambda row: (row["business_id"], row["user_id"]))
    RDD_I = RDD_I.cache()
    BB = RDD_I.groupByKey()
    BB = BB.map(lambda x: (x[0], set(x[1])))
    UL = sorted(RDD_I.map(lambda x: x[1]).distinct().collect())
    n_bfsig = len(UL)
    user_set = set(UL)
    def B_M(users_reviewed, user_set):
        column = list()
        index = 0
        for user in user_set:
            if user in users_reviewed:
                column.append(index)
            index += 1
        return column
    O_mtrx = BB.map(lambda row: (row[0], B_M(row[1], user_set)))
    O_mtrx_data = O_mtrx.collect()
    O_mtrx_data = {item[0] : item[1] for item in O_mtrx_data}
    def B_S(user_list, p, m) :
        SL = list()
        for i in range(1, n_hashes + 1):
            hash_value = list()
            for user in user_list:
                hash_value.append(((i * (((user * 1996 + 21) % 26777) % n_bfsig) + i * (((user * 1996 + 21) % 26777) % n_bfsig) + i * i) % p) % m)
            SL.append(min(hash_value))
        return SL  
    sgn_mtrx = O_mtrx.map(lambda row: (row[0], B_S(row[1], 37307, n_bfsig))) # 37307
    def BCfB(signature):
        bucket_of_bands = list()
        for i in range(bands):
            row_data = signature[1][(rows * i): (rows * (i + 1))]
            bucket_of_bands.append(((i, tuple(row_data)), signature[0]))
        return bucket_of_bands
    BD = sgn_mtrx.flatMap(lambda row: BCfB(row))
    BD = BD.groupByKey().map(lambda row: (row[0], set(row[1]))).filter(lambda row: len(row[1]) > 1)
    
    candidate_pairs = BD.flatMap(lambda row: (combinations(sorted(row[1]), 2))).distinct()
    c = candidate_pairs.collect()
    def J_S(pair):
        set_A = set(O_mtrx_data[pair[0]])
        set_B = set(O_mtrx_data[pair[1]])
        n_union = len(set_A.union(set_B))
        n_intersection = len(set_A.intersection(set_B))
        JS = float(float(n_intersection) / float(n_union))
        return JS
    JSL = candidate_pairs.map(lambda pair: (pair, J_S(pair))).filter(lambda pair: pair[1] >= 0.05).collect()
    JSL = sorted(JSL)

    sc.stop()
    with open(OF, "w+") as file:
       for pair in JSL:
         row_dict = {}
         row_dict["b1"] = pair[0][0]
         row_dict["b2"] = pair[0][1]
         row_dict["sim"] = pair[1]
         json.dump(row_dict, file)
         file.write("\n")
    file.close()
    print ("\nDuration:" + str(time.time() - Tstart))
