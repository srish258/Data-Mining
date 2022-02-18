
from pyspark import SparkConf, SparkContext, StorageLevel
import json
import time
import sys
import math


if __name__ == "__main__":

    IF = sys.argv[1]
    MF = sys.argv[2]
    OF = sys.argv[3]
    Tstart = time.time()
    conf = (SparkConf().setAppName("task2").set("spark.driver.memory", "4g").set("spark.executor.memory", "4g"))
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    model = sc.textFile(MF)
    model = model.map(lambda line: (json.loads(line)["profile_type"],json.loads(line)["profile_id"],json.loads(line)["profile_content"],))
    model = model.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)
    userProfile = model.filter(lambda triple: triple[0] == 'user')
    userProfile = userProfile.map(lambda triple : (triple[1],triple[2]))
    userProfile = userProfile.collectAsMap()
    businessProfile = model.filter(lambda triple: triple[0] == 'business').map(lambda triple : (triple[1],triple[2])).collectAsMap()
    test = sc.textFile(IF)
    def cosine_similarity(pair, businessProfile, userProfile):
        if pair[0] not in businessProfile or pair[1] not in userProfile:
            return 0
        a = set(businessProfile[pair[0]])
        v = set(userProfile[pair[1]])
        return len(a.intersection(v))/(math.sqrt(len(a)) * math.sqrt(len(v)))
    test = test.map(lambda line : (json.loads(line)["business_id"],json.loads(line)["user_id"], cosine_similarity((json.loads(line)["business_id"],json.loads(line)["user_id"]), businessProfile, userProfile)))
    test = test.filter(lambda triple : triple[2] >= 0.01 )
    test = test.collect()
    output = open(OF, "a")
    for triple in test:
        user_id = triple[1]
        business_id = triple[0]
        sim = triple[2]
        content = json.dumps(
            {"user_id": user_id, "business_id": business_id, "sim": sim}
        )
        output.write(content)
        output.write("\n")
    output.close()
    print("Accuracy: " + str(len(test)/58480))
    Tend = time.time()
    print("Duration: ", Tend - Tstart, "s")













