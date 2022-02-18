from pyspark import SparkContext
import sys
import json, operator


if __name__ == "__main__":
    IF = sys.argv[1]
    OF = sys.argv[2]
    partition_type = sys.argv[3]
    n_partitions = int(sys.argv[4])
    n = int(sys.argv[5])
    sc = SparkContext('local[*]', 'task3')
    sc.setLogLevel("ERROR")
    def save_result(results):
        with open(OF, 'w') as w:
            w.write(json.dumps(results))
    results = {}

    input = sc.textFile(IF)

    Rdd_reviews = input.map(lambda review: json.loads(review))\
        .map(lambda item: (item["business_id"], 1))

    if partition_type == "customized":
        Rdd_reviews = Rdd_reviews.partitionBy(n_partitions, lambda x: ord(x[0]))

    results["n_partitions"] = Rdd_reviews.getNumPartitions()

    n_items = Rdd_reviews.glom().map(len).collect()
    results["n_items"] = n_items

    result_rdd = Rdd_reviews.reduceByKey(lambda a,b: a+b).filter(lambda kv: kv[1] > n).collect()
    results["result"] = result_rdd
    print (results)
    save_result(results)