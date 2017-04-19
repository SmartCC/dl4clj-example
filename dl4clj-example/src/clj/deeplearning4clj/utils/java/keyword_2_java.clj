(ns deeplearning4clj.utils.java.keyword-2-java
  (:require [clojure.string :as s]))

(defn keyword-2-java-style
  "转换关键字风格为java风格，去掉每个-连接符并将其后的首字母大写，如to-string转为toString"
  [x]
  (let [[a & b] (s/split x #"-")]
    (apply str a (map s/capitalize b))))

(defmacro transfer-keyword-2-java-function
  "转换keyword为java函数，keyword可用写成clojure风格的形式，如:to-string，也可以写成java风格，如：toString，第二个参数为调用函数的java对象，第三个参数为java函数的参数可以是无参也可以是多个参数"
  [k j-object & forms]
  (let [f-forms (repeatedly (count forms) gensym)
        f `(memfn ~(-> (name k) keyword-2-java-style symbol) ~@f-forms)]
    `(~f ~j-object ~@forms)))

(defmacro keyword-2-java
  "处理函数的输入为多个参数的情况，在使用key-value的形式是，无法使一个key对应多个输入，故将输入组成一个序列，如：(def m (HashMap.)) (keyword-2-java m :put [\"a\" 2])。"
  ([j-object k] `(transfer-keyword-2-java-function ~k ~j-object))
  ([j-object k form]
   (cond
     (nil? form) `(transfer-keyword-2-java-function ~k ~j-object)
     (sequential? form)  `(transfer-keyword-2-java-function ~k ~j-object ~@form)
     :else `(transfer-keyword-2-java-function ~k ~j-object ~form))))

(defmacro ->chain-calls
  "链式调用多个keyword到java函数的转换，keyword和参数必须一一对应，如果参数是个序列，需要在外层包装一个序列，如(HashMap.)，如果直接作参数会被拆开，做参数是需要写成[(HashMap.)]；无参的函数与keyword的对应值为nil"
  [j-object & forms]
  {:pre [(even? (count forms))]}
  (let [forms  (map (fn [[k v]] `(keyword-2-java ~k ~v)) (partition 2 forms))]
    `(-> ~j-object ~@forms)))

(defmacro doto-keyword-2-java
  "doto的keyword-2-java实现版本"
  [j-object & forms]
  {:pre [(even? (count forms))]}
  (let [gx (gensym)]
    `(let [~gx ~j-object]
       ~@(map (fn [[k v]] `(keyword-2-java ~gx ~k ~v)) (partition 2 forms))
       ~gx)))
