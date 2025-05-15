```mermaid
flowchart TD
    A[viewDataRdd] --> B[distinct\nRemove duplicate views]
    B --> C[mapToPair\nReverse key-value positions]
    C --> D[join\nJoin with chapter data]
    D --> E[mapToPair\nDrop chapter id]
    E --> F[reduceByKey\nCount views per user/course]
    F --> G[mapToPair\nDrop userId]

    H[chapterDataRdd] --> I[mapToPair\nMap to courseId, 1]
    I --> J[reduceByKey\nCount chapters per course]

    G --> K[join\nJoin with chapters per course]
    J --> K

    K --> L[mapValues\nCalculate watch percentage and points]
    L --> M[reduceByKey\nSum points per course]
    M --> N[join\nJoin with titles data]
    N --> O[sortByKey\nSort by points]

    subgraph "Step 3: After join with chapter data"
        S3["chapterId    (userId, courseId)
            ---------------------------
            96           (14, 1)
            97           (14, 1)
            96           (13, 1)
            99           (14, 2)
            100          (13, 3)
            ---------------------------"]
    end

    subgraph "Step 5: Count views per user/course"
        S5["(userId, courseId)   count
            ---------------------------
            (14, 1)              2
            (13, 1)              1
            (14, 2)              1
            (13, 3)              1
            ---------------------------"]
    end

    subgraph "Step 6: Drop userId"
        S6["courseId     count
            ---------------------------
            1              2
            1              1
            2              1
            3              1
            ---------------------------"]
    end

    subgraph "Step 7: Chapters per course"
        S7["courseId     chapters
            ---------------------------
            1            3
            2            1
            3            10
            ---------------------------"]
    end

    subgraph "Step 8: Join watched with total"
        S8["courseId     (watched_chapters, total_chapters)
            ---------------------------
            1            (2, 3)
            1            (1, 3)
            2            (1, 1)
            3            (1, 10)
            ---------------------------"]
    end

    subgraph "Step 9: Calculate points"
        S9["courseId     points
            ---------------------------
            1           2
            1           4
            2           10
            3           0
            ---------------------------"]
    end

    subgraph "Step 10: Total points per course"
        S10["courseId     points
            ---------------------------
            1           6
            2           10
            3           0
            ---------------------------"]
    end
```

### If we convert all the input csv files to txt files, then we can run it in (isTestMode = false) <br> but when reading from csv format using jsc.textFile we're getting Hadoop's winutils binary error. <br> Why so ?


The issue occurs because when reading CSV files directly with `textFile()`, Spark uses Hadoop's underlying file system implementation which requires native Windows libraries (winutils.exe) for proper file access.

The reason it works with `.txt` files but not `.csv` files is due to how Windows and Hadoop interact:

1. `.txt` files are treated as plain text files and can often be read using simple file I/O
2. `.csv` files trigger Spark's CSV handling logic which uses Hadoop's file system abstraction

To properly handle CSV files on Windows, you have three options:

1. **Use SparkSession and DataFrame API** (Recommended):
```java
SparkSession spark = SparkSession.builder()
        .appName("spark-big-data-exercise")
        .master("local[*]")
        .getOrCreate();

Dataset<Row> titlesDF = spark.read()
        .option("header", "true")
        .csv("src/main/resources/udemy/virtualPairProgrammers/viewingFigures/titles.csv");

JavaPairRDD<Integer, String> titlesRDD = titlesDF.javaRDD()
        .mapToPair(row -> new Tuple2<>(
            row.getInt(0),
            row.getString(1)
        ));
```

2. **Set up Hadoop environment** properly:
```java
System.setProperty("hadoop.home.dir", "C:\\hadoop");
System.setProperty("hadoop.dll.path", "C:\\hadoop\\bin");
```

3. **Use local file system implementation**:
```java
SparkConf sparkConf = new SparkConf()
        .setAppName("spark-big-data-exercise")
        .setMaster("local[*]")
        .set("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem");
```

The DataFrame API approach is recommended as it handles CSV parsing more robustly and has better Windows compatibility.