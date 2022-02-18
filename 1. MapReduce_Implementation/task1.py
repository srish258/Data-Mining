import sys
from pyspark import SparkContext
import json, operator
S_C = SparkContext('local[*]', 'task1')
S_C.setLogLevel("ERROR")
if __name__ == '__main__':
    IP = sys.argv[1]
    RR = S_C.textFile(IP)
    rrd = RR.map(lambda x: json.loads(x)).cache()
    OP = sys.argv[2]
    output = {}
    CT = rrd.count()
    output["A"] = CT
    Y = sys.argv[4]
    CY = rrd.filter(lambda x: Y in x["date"]).count()
    output["B"] = CY
    CDU = rrd.map(lambda x: x["user_id"]).distinct().count()
    output["C"]=CDU
    M = sys.argv[5]
    TMU = rrd.map(lambda x: (x["user_id"], 1))\
        .reduceByKey(operator.add)\
        .sortBy(lambda x: -x[1])\
        .take(int(M))
    output["D"] = TMU
    SWP = sys.argv[3]
    n = sys.argv[6]
    stopwords = S_C.textFile(SWP).collect()
    output["E"] = rrd.map(lambda review: review["text"].translate({ord(i): None for i in ',.!?:;()[]'}).lower())\
        .flatMap(lambda review: review.split())\
        .filter(lambda w: w and w not in stopwords)\
        .map(lambda w: (w, 1))\
        .reduceByKey(operator.add)\
        .takeOrdered(int(n), key=lambda x: (-x[1], x[0]))
    output["E"] = list(map(lambda x: x[0], output["E"]))
    print(output)
    with open(OP, 'w') as w:
        w.write(json.dumps(output))