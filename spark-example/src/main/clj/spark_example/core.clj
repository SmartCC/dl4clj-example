(ns spark-example.core
  (:require [clojure.string :as s]
            [sparkling.conf :as conf]
            [sparkling.core :as spark]
            [sparkling.destructuring :as s-de])
  (:gen-class :name "sparkExample.WordCount"))

(defn -main
  ""
  [& opts]
  (let [sc (-> (conf/spark-conf)
               (conf/master "local")
               (conf/app-name "wordcount")
               spark/spark-context)]
    (->> (spark/text-file sc "/input/wordcount/wordcount")
         (spark/flat-map (fn [l] (s/split l #" ")))
         (spark/map-to-pair (fn [w] (spark/tuple w 1)))
         (spark/reduce-by-key +)
         (spark/map (s-de/key-value-fn (fn [k v] (str k " appears " v " times."))))
         (spark/save-as-text-file "/output/wordcount/sparkling00"))))
