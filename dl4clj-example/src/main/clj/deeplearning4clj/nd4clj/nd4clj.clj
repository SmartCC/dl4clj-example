(ns deeplearning4clj.nd4clj.nd4clj
  (:import [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.factory Nd4j]))

(defn zeros
  "创建零向量"
  ([columns] (Nd4j/zeros columns))
  ([rows columns] (Nd4j/zeros rows columns)))

(defn *put-scalar*
  "对INDArray赋值，赋值格式为: 坐标 值"
  [^INDArray a & index-&-value]
  {:pre (even? (count index-&-value))}
  (doseq [[idx val] (partition 2 index-&-value)]
    (.putScalar (int-array idx) val)))

(defn dims
  "获取序列构成向量的维度，同时检查其各个维度上的长度是否相等"
  [x]
  (loop [x x dims [(count x)]]
    (cond
      (every? sequential? x) (let [len (count (first x))]
                               (if (every? #(= len (count %)) x)
                                 (recur (mapcat identity x) (conj dims len))
                                 (throw (Exception. "向量维度的宽度不统一"))))
      (some sequential? x) (throw (Exception. "向量维度的深度不统一"))
      :else dims)))

(defn seq-2-ndarray
  "序列转为ndarray"
  [x]
  (Nd4j/create (double-array (flatten x)) (int-array (dims x))))
