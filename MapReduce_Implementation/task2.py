from pyspark import SparkContext
import sys, json, operator

if __name__ == "__main__":
    # parse commandline argument
    RF = sys.argv[1]
    BF = sys.argv[2]
    OF = sys.argv[3]
    result = {}

    def flat(item):
            list = []
            for c in item[1][1]:
                list.append((c, item[1][0]))
            return list

    def filter_RDD(categories):
        list = categories.split(",")
        for i in range(len(list)):
            list[i] = list[i].strip()
        return list

    def get_result_RDD(R_Join):
        R_S = R_Join.reduceByKey(operator.add)
        R_C = R_Join.map(lambda x: (x[0], 1)).reduceByKey(operator.add)
        R_avg = R_S.join(R_C).map(lambda x: (x[0], round(x[1][0] / x[1][1], 1)))
        return R_avg.takeOrdered(n, lambda x: (-x[1], x[0]))

    def read_data(BF, RF):
        Rdd_reviews = sc.textFile(RF)
        Rdd_reviews = Rdd_reviews.map(lambda r: json.loads(r))\
            .map(lambda i: (i["business_id"], i["stars"]))
        Rdd_business = sc.textFile(BF)
        Rdd_business = Rdd_business.map(lambda i: json.loads(i)).filter(lambda i: i["categories"])\
            .map(lambda i: (i["business_id"], filter_RDD(i["categories"])))
        RDD_reviews = Rdd_reviews.partitionBy(Rdd_reviews.getNumPartitions())
        RDD_business = Rdd_business.partitionBy(Rdd_business.getNumPartitions())
        return RDD_reviews,RDD_business
    
    def read_data_wo_spark(BF,RF):
        R = []
        with open(RF, 'r') as fp:
            for line in fp:
                line = json.loads(line)
                R.append((line["business_id"], line["stars"]))
        BC = {}
        with open(BF, 'r') as fp:
            for line in fp:
                line = json.loads(line)
                if line["categories"] == None:
                    continue
                BC[line["business_id"]] = filter_RDD(line["categories"])
        return R,BC

    def category_star(R,BC):
        CS = {}
        for r in R:
            if r[0] in BC:
                for c in BC[r[0]]:
                    if c in CS:
                        CS[c][0] += r[1]
                        CS[c][1] += 1
                    else:
                        CS[c] = [r[1], 1]
        return CS
    def get_result_wo_spark(CS):
        result["result"] = []
        count = 0
        for k, v in sorted(CS.items(), key=lambda item: (-item[1], item[0])):
            result["result"].append((k, v))
            count += 1
            if count == n:
                break
        return result
    def save_file(result):
        with open(OF, 'w') as f:
            json.dump(result, f, sort_keys=True)

    if_spark = sys.argv[4] # only "spark" and "no_spark"
    n = int(sys.argv[5])
    if if_spark == "spark" :
        sc = SparkContext('local[*]', 'task2')
        sc.setLogLevel("ERROR")
        RR,RB = read_data(BF,RF)
        RJ = RR.join(RB).flatMap(flat)
        result["result"] = get_result_RDD(RJ)
    else:
        
        R,BC = read_data_wo_spark(BF,RF)
        
        Categary_S = category_star(R,BC)
        for item in Categary_S.items():
            Categary_S[item[0]] = round(item[1][0] / item[1][1], 1)
        result = get_result_wo_spark(Categary_S)
    print(result)
    save_file(result)

    
        