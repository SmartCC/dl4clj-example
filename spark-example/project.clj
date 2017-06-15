(defproject chunchang/spark-example "0.1.0-SNAPSHOT"
  :description "dl4j examples running on spark"
  :url "https://github.com/SmartCC/dl4clj-example/tree/master/spark-example"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :repositories  [["snapshots-repo" {:url "https://oss.sonatype.org/content/repositories/snapshots"
                                     :snapshots true
                                     :releases false}]]
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [gorillalabs/sparkling "2.0.0"]
                 [org.nd4j/nd4j-native-platform "0.8.0"]
                 [org.deeplearning4j/deeplearning4j-core "0.8.0"]
                 [org.deeplearning4j/deeplearning4j-nlp "0.8.0"]
                 [org.deeplearning4j/deeplearning4j-ui_2.11 "0.8.0"]
                 [com.google.guava/guava "19.0"]
                 [org.datavec/datavec-data-codec "0.8.0"]
                 [org.apache.httpcomponents/httpclient "4.3.5"]
                 [org.deeplearning4j/dl4j-spark_2.11 "0.8.0_spark_2"]]
  :source-paths ["src/main/clj"]
  :test-paths ["src/test/clj"]                                      
  :resource-paths ["src/main/resource"]
  :aot :all)
