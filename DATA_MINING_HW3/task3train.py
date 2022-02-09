import time
import sys
import json
from pyspark import SparkConf, SparkContext
from operator import add
import string
import math
from itertools import combinations

if __name__ == "__main__":
    start = time.time()
    conf = SparkConf().setAppName("Task-3-train").set("spark.executor.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    IF = sys.argv[1]
    MF = sys.argv[2]
    CF_type = sys.argv[3]
    mcr = 3
    reqd_js = 0.01
    n_hashes = 38
    bands = 38
    rows = n_hashes // bands
    RDD_input = sc.textFile(IF, 4)
    RDD_i = RDD_input.map(json.loads).map(lambda row: (row["business_id"], row["user_id"], row["stars"])).cache()

    def Pearson_C(pair, i_dict, j_dict):
        U = set(i_dict.keys()).intersection(set(j_dict.keys()))
        len_U = len(U)
        avg_ri = float(float(sum(i_dict[user] for user in U)) / float(len_U))
        avg_rj = float(float(sum(j_dict[user] for user in U)) / float(len_U))
        Nr = 0
        Dr_1 = 0
        Dr_2 = 0
        for user in U:
            Nr += float(float((i_dict[user] - avg_ri)) * float((j_dict[user] - avg_rj)))
            Dr_1 += float(float((i_dict[user] - avg_ri)) ** 2)
            Dr_2 += float(float((j_dict[user] - avg_rj)) ** 2)
        Dr = float(math.sqrt(Dr_1)) * float(math.sqrt(Dr_2))
        try:
            w_ij = float(Nr) / float(Dr)
        except ZeroDivisionError:
            return float(0)
        return w_ij

    def M_F(pearson_list):
        with open(MF, "w+") as file:
            if CF_type == "item_based":
                for pair in pearson_list:
                    row_dict = {}
                    row_dict["b1"] = pair[0][0]
                    row_dict["b2"] = pair[0][1]
                    row_dict["sim"] = pair[1]
                    json.dump(row_dict, file)
                    file.write("\n")
            elif CF_type == "user_based":
                for pair in pearson_list:
                    row_dict = {}
                    row_dict["u1"] = pair[0][0]
                    row_dict["u2"] = pair[0][1]
                    row_dict["sim"] = pair[1]
                    json.dump(row_dict, file)
                    file.write("\n")
        file.close()
    def FDP(list1, list2):
            len_intersection = len(set(list1).intersection(set(list2)))
            if len_intersection >= 3:
                return True
            else:
                return False
    if CF_type == "user_based":
        print ("\nImplement it bro!!!")
        RDD_U = RDD_i.map(lambda row: ((row[1], row[0]), row[2])).cache()
        RDD_UM = RDD_U.groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda x: (x[0][0], (x[0][1], x[1][-1]))).groupByKey().map(lambda x: (x[0], set(x[1]))).map(lambda x: (x[0], dict((k, v) for k, v in x[1])))
        RDD_UM = RDD_UM.filter(lambda x: len(x[1]) >= 3)
        SB = set(sorted(RDD_i.map(lambda x: x[0]).distinct().collect()))
        Dict_UR = RDD_UM.collectAsMap()
        n_bfsig = len(SB)
        def B_M(business_reviewed, business_set):
            column = list()
            index = 0
            business_reviewed_set = set(business_reviewed.keys())
            for business in business_set:
                if business in business_reviewed_set:
                    column.append(index)
                index += 1
            return column
        U_mtrx = RDD_UM.map(lambda x: (x[0], B_M(x[1], SB)))
        U_mtrx_data = U_mtrx.collectAsMap()
        def B_S(business_list, p, m):
            SL = list()
            for i in range(1, n_hashes + 1):
                hash_value = list()
                for business in business_list:
                    hash_value.append(((i * (((business * 1997 + 15) % 37307) % n_bfsig) + i * (((business * 1997 + 15) % 37307) % n_bfsig) + i * i) % p) % m)
                SL.append(min(hash_value))
            return SL 
        U_sgn_mtrx = U_mtrx.map(lambda x: (x[0], B_S(x[1], 37307, n_bfsig)))
        def BCfB(signature):
            bucket_of_bands = list()
            for i in range(bands):
                row_data = signature[1][(rows * i): (rows * (i + 1))]
                bucket_of_bands.append(((i, tuple(row_data)), signature[0]))
            return bucket_of_bands
        BD = U_sgn_mtrx.flatMap(lambda row: BCfB(row))
        BD = BD.groupByKey().map(lambda row: (row[0], set(row[1]))).filter(lambda row: len(row[1]) > 1)
        CPairs = BD.flatMap(lambda row: (combinations(sorted(row[1]), 2))).distinct()
        
        CPairs = CPairs.filter(lambda x: FDP(U_mtrx_data[x[0]], U_mtrx_data[x[1]]))
        
        def J_S(pair) :
            set_A = set(U_mtrx_data[pair[0]])
            set_B = set(U_mtrx_data[pair[1]])
            n_union = len(set_A.union(set_B))
            n_intersection = len(set_A.intersection(set_B))
            JS = float(float(n_intersection) / float(n_union))
            return JS
        
        RDD_JS = CPairs.map(lambda pair: (pair, J_S(pair))).filter(lambda pair: pair[1] >= reqd_js).map(lambda x: x[0])
        RDD_UPerson = RDD_JS.map(lambda pair: (pair, Pearson_C(pair, Dict_UR[pair[0]], Dict_UR[pair[1]]))).filter(lambda x: x[1] > 0)
        UPL = sorted(RDD_UPerson.collect(), key = lambda x: (x[0][0], -x[1]), reverse = False)
        
        M_F(UPL)

    else:
        print ("\nImplement it bro!!!")
        
        def CBRD(BRL):
            BRD = dict()
            for rating in BRL:
                BRD[rating[0]] = rating[1]
            return BRD
        def weighted_average(ratings_list):
            return ratings_list[-1]
        RDD_BR = RDD_i.map(lambda x: ((x[0], x[1]), x[2])).groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda x: (x[0][0], (x[0][1], weighted_average(x[1])))).groupByKey().distinct().map(lambda x: (x[0], set(x[1]))).map(lambda x: (x[0], CBRD(x[1])))
        RDD_BR = RDD_BR.filter(lambda x: len(x[1]) >= 3)
        Dict_BR = RDD_BR.collectAsMap()
        RDD_BI = RDD_BR.map(lambda x: x[0])
        RDD_BPair = RDD_BI.cartesian(RDD_BI).filter(lambda x: x[0] < x[1]).filter(lambda x: FDP(Dict_BR[x[0]].keys(), Dict_BR[x[1]].keys()))
        RDD_BPerson = RDD_BPair.map(lambda pair: (pair, Pearson_C(pair, Dict_BR[pair[0]], Dict_BR[pair[1]]))).filter(lambda x: x[1] > 0)
        BPL = sorted(RDD_BPerson.collect(), key = lambda x: (x[0][0], -x[1]), reverse = False)
        
        M_F(BPL)    
        
    sc.stop()
    print ("\nDuration:" + str(time.time() - start))