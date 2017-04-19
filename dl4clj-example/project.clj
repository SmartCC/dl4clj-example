(defproject dl4clj-example "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.nd4j/nd4j-native-platform "0.8.0"]
                 [org.deeplearning4j/deeplearning4j-core "0.8.0"]
                 [org.deeplearning4j/deeplearning4j-nlp "0.8.0"]
                 [com.google.guava/guava "19.0"]
                 [org.datavec/datavec-data-codec "0.8.0"]
                 [org.jfree/jcommon "1.0.23"]]
  :source-paths ["src/clj"]
  :test-paths ["test/clj"]
  :aot :all)
