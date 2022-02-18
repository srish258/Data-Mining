from pyspark import SparkContext, SparkConf
import time
import json
import re
import math
import sys
OVERALL_AVG = 3.5



def get_predictions(ub_score_list, model, avg_d):
    """Use the Pearson Similarity Scoring for Predictions."""
    u1, b1 = ub_score_list[0][0], ub_score_list[0][1]
    possible_neighbors = []
    if CF_type == "item_based":
        for bus_sim in ub_score_list[1]:
            bus_, rating = bus_sim[0], bus_sim[1]
            combo = (b1, bus_)
            if combo in model:
                sim_ = model[combo]
                possible_neighbors.append((rating, sim_))
                continue

            else:
                combo = (bus_, b1)
                if combo in model:
                    sim_ = model[combo]
                    possible_neighbors.append((rating, sim_))
                    continue

        k_neighbors = sorted(possible_neighbors, key=lambda x: -x[1])[:5]
        num_ = sum([rating_sim[0] * rating_sim[1] for rating_sim in k_neighbors])
        if num_ == 0:
            if b1 in avg_d:
                return (u1, b1, avg_d[b1])
            else:
                return (u1, b1, OVERALL_AVG)

        denom_	= sum([abs(rating_sim[1]) for rating_sim in k_neighbors])
        if denom_ == 0:
            if b1 in avg_d:
                return (u1, b1, avg_d[b1])
            else:
                return (u1, b1, OVERALL_AVG)

        rating = num_/denom_
        return (u1, b1, rating)

    else:
        u1, b1 = ub_score_list[0][1], ub_score_list[0][0]
        for user_sim in ub_score_list[1]:
            user_, rating = user_sim[0], user_sim[1]
            combo = (u1, user_)
            if combo in model:
                sim_ = model[combo]
                possible_neighbors.append((rating, avg_d[user_], sim_))
                continue

            else:
                combo = (user_, u1)
                if combo in model:
                    sim_ = model[combo]
                    possible_neighbors.append((rating, avg_d[user_], sim_))
                    continue

        k_neighbors = sorted(possible_neighbors, key=lambda x: -x[2])[:5]
        num_ = sum([(rating_sim[0] - rating_sim[1]) * rating_sim[2] for rating_sim in k_neighbors])
        if num_ == 0:
            if u1 in avg_d:
                return (u1, b1, avg_d[u1])
            else:
                return (u1, b1, OVERALL_AVG)

        denom_ = sum([abs(rating_sim[2]) for rating_sim in k_neighbors])
        if denom_ == 0:
            if u1 in avg_d:
                return (u1, b1, avg_d[u1])
            else:
                return (u1, b1, OVERALL_AVG)

        rating = avg_d[u1] + num_ / denom_
        return (u1, b1, rating)

if __name__ == "__main__":
    start = time.time()

    UAF = "../resource/asnlib/publicdata/user_avg.json"
    BAF = "../resource/asnlib/publicdata/business_avg.json"
    IFtrain = sys.argv[1]
    IFtest = sys.argv[2]
    IMF = sys.argv[3]
    OF = sys.argv[4]
    CF_type = sys.argv[5]
    conf = SparkConf().set("spark.executor.memory", "8g").set("spark.driver.memory", "8g")
    sc = SparkContext(conf=conf)
    test_rdd = sc.textFile(IFtest)
    train_rdd = sc.textFile(IFtrain)
    model_rdd = sc.textFile(IMF)
    B_avg_rdd = sc.textFile(BAF)
    U_avg_rdd = sc.textFile(UAF)
    if CF_type == "item_based":
        
        user_business_test_rdd = test_rdd.map(lambda x: json.loads(x)).map(lambda u_b: (u_b["user_id"], u_b["business_id"])).persist()
        user_bus_stars_train_rdd = train_rdd.map(lambda x: json.loads(x)).map(lambda u_b_s: (u_b_s["user_id"], u_b_s["business_id"], u_b_s["stars"])).persist()
        b1_b2_sim_model_rdd = model_rdd.map(lambda x: json.loads(x)).map(lambda b1_b2_s: ((b1_b2_s["b1"], b1_b2_s["b2"]), b1_b2_s["sim"])).collectAsMap()
        bus_avg_d = B_avg_rdd.map(lambda x: json.loads(x)).map(lambda x: dict(x)).flatMap(lambda x: [(key, val) for key, val in x.items()]).collectAsMap()
        user_business_ratings = user_bus_stars_train_rdd.map(lambda u_b_s: (u_b_s[0], (u_b_s[1], u_b_s[2]))).groupByKey().map(lambda u_bsL: (u_bsL[0], list(u_bsL[1])))
        user_business_rating = user_business_test_rdd.leftOuterJoin(user_business_ratings).map(lambda x: ((x[0], x[1][0]), x[1][1])).filter(lambda ub_brL: ub_brL[1] != None).groupByKey().map(lambda ub_brLL: (ub_brLL[0], [item for sublist in ub_brLL[1] for item in sublist])).map(lambda ub_brL: get_predictions(ub_brL, b1_b2_sim_model_rdd, bus_avg_d)).collect()

    else:
        
        bus_user_test_rdd = test_rdd.map(lambda x: json.loads(x)).map(lambda u_b: (u_b["business_id"], u_b["user_id"])).persist()

        user_bus_stars_train_rdd = train_rdd.map(lambda x: json.loads(x)).map(lambda u_b_s: (u_b_s["user_id"], u_b_s["business_id"], u_b_s["stars"])).persist()

        u1_u2_sim_model_rdd = model_rdd.map(lambda x: json.loads(x)).map(lambda u1_u2_s: ((u1_u2_s["u1"], u1_u2_s["u2"]), u1_u2_s["sim"])).collectAsMap()

        
        user_avg_d = U_avg_rdd.map(lambda x: json.loads(x)).map(lambda x: dict(x)).flatMap(lambda x: [(key, val) for key, val in x.items()]).collectAsMap()

        # business with all of it's user ratings: {b1: [(u1, r1), (b2, r2), ...)]
        bus_user_ratings = user_bus_stars_train_rdd.map(lambda u_b_s: (u_b_s[1], (u_b_s[0], u_b_s[2]))).groupByKey().map(lambda b_usL: (b_usL[0], list(b_usL[1])))

        user_business_rating = bus_user_test_rdd.leftOuterJoin(bus_user_ratings).map(lambda x: ((x[0], x[1][0]), x[1][1])).filter(lambda bu_urL: bu_urL[1] != None).groupByKey().map(lambda bu_urLL: (bu_urLL[0], [item for sublist in bu_urLL[1] for item in sublist])).map(lambda bu_urL: get_predictions(bu_urL, u1_u2_sim_model_rdd, user_avg_d)).collect()


    with open(OF, "w") as w:
        for u1_b1_sim in user_business_rating:
            w.write(json.dumps({"user_id": u1_b1_sim[0], "business_id": u1_b1_sim[1], "stars": u1_b1_sim[2]}) + '\n')

    w.close()
    end = time.time()
    print("Duration:", round((end-start), 2))





