import java.io._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scala.util.parsing.json.JSONObject
import scala.collection.mutable.{HashMap, Map, ListBuffer}
import scala.util.Random
import scala.util.matching.Regex



object task1{

	def cal_euclid_distacnce(point1: Array[Double], point2: Array[Double]): Double ={
		var distance = 0.0
		for (i <- 0 to point1.size-1) {
			distance += (point1(i) - point2(i))*(point1(i) - point2(i))
		}
		math.sqrt(distance)

	}

	def cal_nearest_cluster(point: Int, point_location: Array[Double], centeroids: Array[Tuple2[Int,Array[Double]]]): Tuple2[Int, Int]={
		var min_distance = Double.MaxValue
		var nearest_cluster = 0
		for(center <- centeroids){
			val center_location = center._2
			val distance = cal_euclid_distacnce(point_location, center_location)
			if(min_distance > distance) {
				min_distance = distance
				nearest_cluster = center._1
			}
		}
		Tuple2(nearest_cluster, point)
	}

	def select_centroids(dataset : Array[Int], cluster_num: Int, location_dict: Map[Int, Array[Double]]): Array[Tuple2[Int, Array[Double]]] ={
		var count = 0
		var centroids = List(Tuple2(count, location_dict.apply(Random.shuffle(dataset.toList).apply(0))))
		var max_min_distance : Map[Int, Double] = Map()
		for (point<- dataset){
			max_min_distance += (point -> cal_euclid_distacnce(location_dict.apply(point), centroids.apply(0)._2))
		}
		while (count < cluster_num -1){
			count += 1
			val next_sample = location_dict.apply(max_min_distance.maxBy(x => x._2)._1)

			centroids ::= Tuple2(count, next_sample)

			for(point <- dataset){
				val next_distance = cal_euclid_distacnce(next_sample, location_dict.apply(point))
				max_min_distance.put(point, math.min(max_min_distance.apply(point), next_distance))
			}
		}
		centroids.sortBy(x=>x._1).toArray
	}

	def cal_centroids(cluster: Array[Int], location_dict: Map[Int, Array[Double]], d: Int): Array[Double]={
		val new_centroid = new Array[Double](d)
		val N = cluster.size
		for (point <- cluster){
			val location = location_dict.apply(point)
			for(i<- 0 to d-1){
				new_centroid(i) += (location.apply(i)/N)
			}
		}
		new_centroid
	}

	def K_means(dataset: RDD[Int], cluster_num: Int, location_dict: Map[Int, Array[Double]]): Array[Array[Int]] ={
		val cluster_dict: Map[Int, Int] = Map()
		val point_inf = dataset.collect()
		for (point <- point_inf){
			cluster_dict += (point -> point)
		}

		var centroids = select_centroids(point_inf, cluster_num, location_dict)
		val d = centroids(0)._2.size
		val point_num = point_inf.size
		var change = point_num
		var times = 0

		while (change > point_num/50 && change > 100 && times < 10){
			change = 0
			val clusters = dataset.map(x => cal_nearest_cluster(x, location_dict.apply(x), centroids))
			centroids = clusters.groupByKey().mapValues(x => cal_centroids(x.toArray, location_dict, d)).collect()
			for(point <- clusters.collect()){
				if (cluster_dict.apply(point._2) != point._1){
					change += 1
				}
				cluster_dict.put(point._2, point._1)
			}
			times += 1
		}
		val ans = dataset.map(x => cal_nearest_cluster(x, location_dict.apply(x), centroids)).groupByKey().map(x=>x._2.toArray).collect()
		ans
	}

	def cal_std(cluster_inf: Tuple3[Int, Array[Double], Array[Double]]): Array[Double] ={
		val N = cluster_inf._1
		val SUM = cluster_inf._2
		val SUMSQ = cluster_inf._3
		val d = SUM.size
		val std = new Array[Double](d)
		for (i<- 0 to d-1){
			std(i) = math.sqrt(math.max(1e-10, (SUMSQ(i)/N) - (SUM(i)/N)*(SUM(i)/N)))
		}
		std
	}

	def cal_statistics(cluster: Array[Int], location_dict: Map[Int, Array[Double]], d: Int): (Int, Array[Double], Array[Double]) ={
		val SUM = new Array[Double](d)
		val SUMSQ = new Array[Double](d)
		var N = 0
		for(point <- cluster){
			val location = location_dict.apply(point)
			for(i <- 0 to d-1) {
				SUM(i) += location(i)
				SUMSQ(i) += location(i)*location(i)
			}
			N += 1
		}
		(N, SUM, SUMSQ)
	}

	def cal_Mahalanobis_distance(point_location: Array[Double], cluster_inf: Map[Int, (Int, Array[Double], Array[Double])], M_threshold: Double): Int ={
		var belong : Int = -1
		var min_distance = Double.MaxValue
		for(i <- 0 to cluster_inf.size-1){
			if (cluster_inf.apply(i)._1 > 1){
				val center = cluster_inf.apply(i)._2.map(x=> x/cluster_inf.apply(i)._1)
				val std = cal_std(cluster_inf.apply(i))
				var distance = 0.0
				for(j <- 0 to std.size-1){
					distance += ((point_location(j)- center(j))/std(j))*((point_location(j)- center(j))/std(j))
				}
				distance = math.sqrt(distance)
				if (distance < M_threshold && distance < min_distance){
					min_distance = distance
					belong = i
				}
			}
		}

		belong
	}

	def update_statistics(point: Array[Double], inf: (Int, Array[Double], Array[Double])) : (Int, Array[Double], Array[Double]) ={
		val N = inf._1 + 1
		val SUM = inf._2
		val SUMSQ = inf._3

		for( i <- 0 to point.size-1){
			SUM(i) += point(i)
			SUMSQ(i) += point(i)*point(i)
		}
		(N, SUM, SUMSQ)
	}

	def main(args: Array[String]): Unit = {
		val start_time = System.currentTimeMillis()
		val path = args.apply(0)
		val c_num = args.apply(1).toInt
		val cluster_res = args.apply(2)
		val inter_data_file = args.apply(3)
		val file_dict = new File(path)
		val file_list = file_dict.listFiles()
		val conf = new SparkConf().setAppName("inf553").setMaster("local[4]")
		val sc = new SparkContext(conf)
		sc.setLogLevel("WARN")
		var inter_data = new ListBuffer[Array[String]]
		val DS_inf : Map[Int, (Int, Array[Double], Array[Double])] = Map()
		val CS_inf : Map[Int, (Int, Array[Double], Array[Double])] = Map()
		val CS_cluster : Map[Int, List[Int]] = Map()
		val res_dict =  new HashMap[String, String]()
		var RS_cluster : Map[Int, Array[Double]] = Map()
		var RS_point : List[Int] = List()
		var DS_count, RS_count, CS_count, DS_cluster_num, CS_cluster_num, d= 0
		var M_threshold: Double = 0
		inter_data += Array("round_id", "nof_cluster_discard", "nof_point_discard", "nof_cluster_compression", "nof_point_compression", "nof_point_retained")
		var cur_index = 1
		val re = new Regex(",")
		for(file <- file_list){
			println(cur_index +"  "+ (System.currentTimeMillis()-start_time)/1000)
			val data = sc.textFile(file.toString).map(x=>re.split(x)).map(x=>(x.apply(0).toInt, x.drop(1).toList.map(x=>x.toDouble).toArray))
			val location = data.collect()
			d = location.apply(0)._2.size
			for(item <-location){
				res_dict.put(item._1.toString, "-1")
			}

			if (cur_index == 1){
				val cur_location_dict : Map[Int, Array[Double]] = Map()
				for(item <-location){
					cur_location_dict += (item._1 ->item._2)
				}
				val dataset = data.map(x=> x._1)
				val sample_points = dataset.sample(false, 0.1)
				val remain_points = dataset.subtract(sample_points)
				val K_means_res = K_means(sample_points, c_num, cur_location_dict)
				for(cluster <- K_means_res){
					DS_inf += (DS_cluster_num -> cal_statistics(cluster, cur_location_dict, d))
					DS_count += cluster.size
					for(point <- cluster){
						res_dict.put(point.toString, DS_cluster_num.toString)
					}
					DS_cluster_num += 1
				}
				val remain_cluster = K_means(remain_points, c_num*5, cur_location_dict)
				for(cluster <- remain_cluster) {
					if (cluster.size > 1) {
						CS_inf += (CS_cluster_num -> cal_statistics(cluster, cur_location_dict, d))
						CS_cluster += (CS_cluster_num -> cluster.toList)
						CS_cluster_num += 1
						CS_count += cluster.size
					}
					else {
						RS_cluster += (cluster.apply(0) -> cur_location_dict.apply(cluster.apply(0)))
						RS_point +:= cluster.apply(0)
						RS_count += 1
					}
				}

			}
			else{
				val point_inf = data.collect()
				M_threshold = 2*math.sqrt(d)
				for(point <- point_inf){
					var cluster_num = cal_Mahalanobis_distance(point._2, DS_inf, M_threshold)
					if (cluster_num != -1){
						DS_count += 1
						res_dict.put(point._1.toString, cluster_num.toString)
						DS_inf.put(cluster_num,update_statistics(point._2, DS_inf.apply(cluster_num)))
					}
					else{

						cluster_num = cal_Mahalanobis_distance(point._2, CS_inf, M_threshold)
						if (cluster_num != -1) {
							CS_count += 1
							CS_cluster.apply(cluster_num) ::= point._1
							CS_inf.put(cluster_num,update_statistics(point._2, CS_inf.apply(cluster_num)))
						}
						else {
							RS_cluster += (point._1 -> point._2)
							RS_point ::= point._1
							RS_count += 1
						}
					}
				}
				if (RS_point.size > c_num*50 && RS_point.size > res_dict.size/100){
					println("running....")
					val RS_rdd = sc.parallelize(RS_point)
					val K_means_on_RS = K_means(RS_rdd, c_num*5, RS_cluster)
					var new_RS_cluster : Map[Int, Array[Double]] = Map()
					var new_RS_point : List[Int] = List()

					for(cluster <- K_means_on_RS){
						if (cluster.size > 1 ){
							val cluster_centroid = cal_centroids(cluster, RS_cluster, d)
							val cluster_num = cal_Mahalanobis_distance(cluster_centroid, CS_inf, M_threshold)
							CS_count += cluster.size
							if (cluster_num == -1){
								CS_cluster += (CS_cluster_num -> cluster.toList)
								CS_inf += (CS_cluster_num -> cal_statistics(cluster, RS_cluster, d))
								CS_cluster_num += 1
							}
							else{
								CS_cluster.put(cluster_num, CS_cluster.apply(cluster_num):::cluster.toList)
								for(point <- cluster){
									CS_inf.put(cluster_num, update_statistics(RS_cluster.apply(point) ,CS_inf.apply(cluster_num)))
								}
							}

						}
						else{
							new_RS_point :::= cluster.toList
							new_RS_cluster += (cluster.apply(0) -> RS_cluster.apply(cluster(0)))
						}
					}
					RS_cluster = new_RS_cluster
					RS_point = new_RS_point
					RS_count = RS_point.size
				}
			}
			inter_data += Array(cur_index.toString, DS_cluster_num.toString, DS_count.toString, CS_cluster_num.toString, CS_count.toString, RS_count.toString)
			cur_index += 1
		}

		for(cluster_index <- 0 to CS_cluster.size-1){
			val cs_cluster_inf = CS_inf.apply(cluster_index)
			val cs_center = cs_cluster_inf._2.map(x=> x/cs_cluster_inf._1)
			val merge_cluster_num = cal_Mahalanobis_distance(cs_center, DS_inf, M_threshold)
			if (merge_cluster_num == -1){
				for(point <- CS_cluster.apply(cluster_index)){
					res_dict.put(point.toString, DS_cluster_num.toString)
				}
				DS_cluster_num += 1
			}
			else{
				for(point<- CS_cluster.apply(cluster_index)){
					res_dict.put(point.toString, merge_cluster_num.toString)
				}
			}
		}
		val json_output = JSONObject(res_dict.toMap).toString()
		val w_1 = new PrintWriter(new File(cluster_res))
		w_1.write(json_output)
		w_1.flush()
		w_1.close()
		val writer = new BufferedWriter(new FileWriter(inter_data_file))
		for(line <- inter_data){
			writer.write(line.mkString(",")+"\n")

		}
		writer.flush()
		writer.close()
		println("Total Duration:" +" "+(System.currentTimeMillis()-start_time)/1000)

	}


}
