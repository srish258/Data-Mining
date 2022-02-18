
from pyspark import SparkConf, SparkContext, StorageLevel
import re
import json
import time
import sys



if __name__ == "__main__":
    conf = (
        SparkConf()
        .setAppName("task2")
        .set("spark.driver.memory", "4g")
        .set("spark.executor.memory", "4g")
    )
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    IF = sys.argv[1]
    MF = sys.argv[2]
    SWF = sys.argv[3]

    TReview = sc.textFile(IF)
    sw = sc.textFile(SWF).collect()
    
    Tstart = time.time()
    comment = TReview.map(lambda line: (json.loads(line)["business_id"],json.loads(line)["text"].lower(),))
    comment = comment.reduceByKey(lambda a, b: a + " " + b)
    comment = comment.map(lambda pair: (pair[0],re.split(r"\\[a-z]|\s|[!'\"#$%&()*+,\-./:;<=>?@\[\]^_`{|}~\\]", pair[1]),))
    def swF(value, stopwords):
        ans = []
        for word in value:
            if (
                word not in stopwords
                and word != ""
                and re.match(r"^(\d+|[a-z])$", word) is None
            ):
                ans.append(word)
        return ans
    comment = comment.mapValues(lambda value: swF(value, sw))
    comment.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)

    wC = comment.flatMap(lambda x: list(map(lambda word: (word, 1), x[1])))
    wC = wC.reduceByKey(lambda a, b: a + b)
    wC.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)

    TWn = wC.map(lambda x: ("count", x[1]))
    TWn = TWn.reduceByKey(lambda a, b: a + b)
    TWn = int(TWn.collect()[0][1])
    
    rareword_threshold = int(TWn * 1e-6)
    rarewords = wC.filter(lambda x: x[1] <= rareword_threshold)
    rarewords = rarewords.collectAsMap()
    def rwF(value, rarewords):
        ans = []
        for word in value:
            if word not in rarewords:
                ans.append(word)
        return ans
    comment = comment.mapValues(lambda value: rwF(value, rarewords))
    comment.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)
    def gwC(x):
        words = x[1]
        hashmap = {}
        ans = []
        for word in words:
            if word not in hashmap:
                hashmap[word] = 0
            hashmap[word] = hashmap[word] + 1
        for word in hashmap:
            ans.append((word, [hashmap[word]]))  # NAN
        return ans
    max_frequency = comment.flatMap(lambda x: gwC(x))
    max_frequency = max_frequency.reduceByKey(lambda a, b: a + b)
    max_frequency = max_frequency.mapValues(lambda value: max(value))
    max_frequency = max_frequency.collectAsMap()
    def makeTf(x, max_frequency):
        k = x[0]
        v = {}
        words = x[1]
        hashmap = {}
        for word in words:
            if word not in hashmap:
                hashmap[word] = 0
            hashmap[word] = hashmap[word] + 1
        for word in hashmap:
            v[word] = (hashmap[word]) / max_frequency[word]
        return (k, v)
    tf = comment.map(lambda x: makeTf(x, max_frequency))
    def makeIdf(x):
        ans = []
        words = x[1]
        s = {}
        for word in words:
            if word not in s:
                s[word] = 0
        for word in s:
            ans.append((word, 1))
        return ans
    N = comment.count()
    idf = comment.flatMap(lambda x: makeIdf(x))
    idf = idf.reduceByKey(lambda a, b: a + b)
    idf = idf.map(lambda pair: (pair[0], N / pair[1]))
    idf = idf.collectAsMap()
    
    def makeTfIdf(value, idf):
        for word in value:
            value[word] = value[word] * idf[word]
        value = dict(sorted(value.items(), key=lambda item: -item[1])[:200])
        return value
    tfIdf = tf.mapValues(lambda value: makeTfIdf(value, idf))

    vector = tfIdf.flatMap(lambda pair: pair[1].keys())
    vector = vector.distinct()
    vector = vector.zipWithIndex()
    vector = vector.collectAsMap()
    def wordsToIndex(value, vector):
        ans = []
        for word in value:
            ans.append(vector[word])
        return set(ans)
    documentProfile = tfIdf.mapValues(lambda value: wordsToIndex(value, vector)).collectAsMap()

    userProfile = TReview.map(lambda line: (json.loads(line)["user_id"], [json.loads(line)["business_id"]]))
    userProfile = userProfile.reduceByKey(lambda a, b: list(set(a + b)))
    def unionIndex(value, documentProfile):
        ans = set()
        for bid in value:
            ans = ans.union(documentProfile[bid])
        return ans
    userProfile = userProfile.mapValues(lambda value: unionIndex(value, documentProfile))
    userProfile = userProfile.collectAsMap()

    model = open(MF, "a")
    for bid in documentProfile:
        v = documentProfile[bid]
        v = list(v)
        v.sort()
        content = json.dumps(
            {"profile_type": "business", "profile_id": bid, "profile_content": tuple(v)}
        )
        model.write(content)
        model.write("\n")

    for uid in userProfile:
        v = userProfile[uid]
        v = list(v)
        v.sort()
        content = json.dumps(
            {"profile_type": "user", "profile_id": uid, "profile_content": tuple(v)}
        )
        model.write(content)
        model.write("\n")
    model.close()

    Tend = time.time()
    print("Duration: ", Tend - Tstart, "s")