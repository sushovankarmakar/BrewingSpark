package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

public class _8_BigDataExercise {

    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf()
                .setAppName("spark-big-data-exercise")
                .setMaster("local[*]");

        try (JavaSparkContext jsc = new JavaSparkContext(sparkConf)) {

            // running it on test mode only (isTestMode = true)
            // so that it is workable BOTH on windows and Mac
            // while making test mode false, it starts reading from file hadoop and in windows, it starts throwing error.

            // tried but didn't work,
            // tried adding Hadoop's winutils binary to my system.
            // Create a directory (e.g., C:\hadoop\bin) and add the winutils.exe file there
            // System.setProperty("hadoop.home.dir", "C:\\hadoop");

            // one more hack is that,
            // if we convert all the input csv files to txt files,
            // then we can run it in (isTestMode = false) i.e. we can read from a file system only.
            // explanation to this has been added into _8_BigDataExercise.md

            boolean isTestMode = true;

            JavaPairRDD<Integer, Integer> chapterDataRdd = setUpChapterDataRdd(jsc, isTestMode);
            JavaPairRDD<Integer, Integer> viewDataRdd = setUpViewDataRdd(jsc, isTestMode);
            JavaPairRDD<Integer, String> titlesDataRdd = setUpTitlesDataRdd(jsc, isTestMode);

            // I WAS DOING WRONG.
            // the reduce function (val1, val2) -> 1 + 1 always returns 2 regardless of the input values.
            // It doesn't properly accumulate the count.
            // first, do mapToPair and then reduceByKey:
            /*JavaPairRDD<Integer, Integer> javaPairRDD = chapterDataRdd
                    .mapToPair(pair -> new Tuple2<>(pair._2, pair._1));*/

            JavaPairRDD<Integer, Integer> chaptersPerCourse = chapterDataRdd
                    .mapToPair(pair -> new Tuple2<>(pair._2, 1))
                    .reduceByKey(Integer::sum);

            System.out.println("the number of chapters on that course.");
            chaptersPerCourse.foreach(val ->
                    System.out.println("Course no. " + val._1 + " has " + val._2 + " chapters.")
            );

            // find out which course has how many points.

            JavaPairRDD<Integer, Long> data = viewDataRdd
                    .distinct() // 1. Removing Duplicate Views
                    .mapToPair(val -> new Tuple2<>(val._2, val._1)) // 2. reverse the position of key and value so that we can join
                    .join(chapterDataRdd) // 3. join so that we ger chapter ids
                    .mapToPair(pair -> new Tuple2<>(pair._2, 1L)) // 4. drop the chapter id, not needed
                    .reduceByKey(Long::sum) // 5. count views for each user per course
                    .mapToPair(pair -> new Tuple2<>(pair._1._2, pair._2)) // 6. drop the userId, not needed
                    .join(chaptersPerCourse) // 8. joining with another RDD
                    .mapValues(row -> { // 9. deciding watch percentage and distributing points

                        long chapterWatched = row._1;
                        int chapterTotal = row._2;

                        double watchPercentage = (double) chapterWatched / chapterTotal;

                        long points = 0L;
                        if (watchPercentage > 0.9)
                            points = 10L;        // as per business rules, we're distributing points
                        else if (watchPercentage > 0.5)
                            points = 4L;
                        else if (watchPercentage > 0.25)
                            points = 2L;

                        return points;
                    })
                    .reduceByKey(Long::sum); // 10. calculating total points of each course

            //data.foreach(val -> System.out.println(val + " "));

            // EXPLANATION for each step :
            // --------------------------

            // after 3rd step : RDD represents, view of a course of : JavaPairRDD<Integer, Tuple2<Integer, Integer>>
            // chapterId    (userId, courseId)
            // ---------------------------
            // 96           (14, 1)
            // 97           (14, 1)
            // 96           (13, 1)
            // 99           (14, 2)
            // 100          (13, 3)
            // ---------------------------
            // chapterId 96 has been seen by userId 14 which is a part of courseId

            // after 4th step : drop the chapter id : JavaPairRDD<Tuple2<Integer, Integer>, Long>
            // (userId, courseId)   count
            // ---------------------------
            // (14, 1)              1
            // (14, 1)              1
            // (13, 1)              1
            // (14, 2)              1
            // (13, 3)              1
            // ---------------------------

            // 5. count views for each user per course : JavaPairRDD<Tuple2<Integer, Integer>, Long>
            // (userId, courseId)   count
            // ---------------------------
            // (14, 1)              2
            // (13, 1)              1
            // (14, 2)              1
            // (13, 3)              1
            // ---------------------------
            // userId 14 has viewed 2 chapters fo courseId 1

            // 6. drop the userId, not needed: JavaPairRDD<Integer, Long> : this data presents for each course how many unique chapters has been watched
            //  courseId     count
            // ---------------------------
            //  1              2
            //  1              1
            //  2              1
            //  3              1

            // 7. find how many chapters each course contains - this data comes from chaptersPerCourse RDD
            // ---------------------------
            // courseId     chapters
            // ---------------------------
            // 1            3
            // 2            1
            // 3            10
            // ---------------------------
            // Course no. 2 has 1 chapter.
            // Course no. 1 has 3 chapters.
            // Course no. 3 has 10 chapters.

            // 8. Join 8th step with 6th step
            // courseId     (watched_chapters, total_chapters)
            // ---------------------------
            // 1            (2, 3)
            // 1            (1, 3)
            // 2            (1, 1)
            // 3            (1, 10)
            // ---------------------------
            // courseId 1 has 3 chapters, but one unique user has seen 2 chapters
            // courseId 1 has 3 chapters, but another unique user has seen 1 chapters

            // 9. deciding watch percentage and distributing points as per business logic : JavaPairRDD<Integer, Long>
            // courseId     points
            // ---------------------------
            //  1           2
            //  1           4
            //  2           10
            //  3           0
            // courseId 1 got 2 points because one unique user watched 25% - 50% of the entire course
            // courseId 1 got 4 points because one unique user watched 50% - 90% of the entire course
            // courseId 2 got 10 points because one unique user watched more than 90% of the entire course
            // courseId 3 got 0 points because one unique user watched less than 25% of the entire course

            // 10. calculating total points achieved by each course so far : JavaPairRDD<Integer, Long>
            // courseId     points
            // --------------------------- - FINAL OUTPUT
            //  1           6
            //  2           10
            //  3           0
            // courseId 1 got total 6 points because two unique users who contributed to 2 points and 4 points
            // courseId 2 got total 10 points
            // courseId 3 got total 0 points
            // ---------------------------

            data.join(titlesDataRdd) // JOINING course id with titles
                    .mapToPair(row -> new Tuple2<>(row._2._1, row._2._2))
                    .sortByKey(false) // 11. sorting based on total points.
                    .collect()
                    .forEach(val -> System.out.println(val + " "));
        }
    }

    private static JavaPairRDD<Integer, Integer> setUpChapterDataRdd(JavaSparkContext jsc, boolean isTestMode) {

        if (isTestMode) {
            List<Tuple2<Integer, Integer>> rawChapterData = new ArrayList<>();
            // chapterId, courseId
            rawChapterData.add(new Tuple2<>(96, 1));
            rawChapterData.add(new Tuple2<>(97, 1));
            rawChapterData.add(new Tuple2<>(98, 1));
            rawChapterData.add(new Tuple2<>(99, 2));
            rawChapterData.add(new Tuple2<>(100, 3));
            rawChapterData.add(new Tuple2<>(101, 3));
            rawChapterData.add(new Tuple2<>(102, 3));
            rawChapterData.add(new Tuple2<>(103, 3));
            rawChapterData.add(new Tuple2<>(104, 3));
            rawChapterData.add(new Tuple2<>(105, 3));
            rawChapterData.add(new Tuple2<>(106, 3));
            rawChapterData.add(new Tuple2<>(107, 3));
            rawChapterData.add(new Tuple2<>(108, 3));
            rawChapterData.add(new Tuple2<>(109, 3));

            return jsc.parallelizePairs(rawChapterData);
        }

        return jsc.textFile("src/main/resources/udemy/virtualPairProgrammers/viewingFigures/chapters.csv")
                .mapToPair(line -> {
                    String[] columns = line.split(",");
                    return new Tuple2<>(
                            Integer.valueOf(columns[0]),
                            Integer.valueOf(columns[1])
                    );
                });
    }

    private static JavaPairRDD<Integer, Integer> setUpViewDataRdd(JavaSparkContext jsc, boolean isTestMode) {

        if (isTestMode) {
            List<Tuple2<Integer, Integer>> rawViewData = new ArrayList<>();

            // userId, chapterId, dateAndTime
            rawViewData.add(new Tuple2<>(14, 96));
            rawViewData.add(new Tuple2<>(14, 97));
            rawViewData.add(new Tuple2<>(13, 96));
            rawViewData.add(new Tuple2<>(13, 96));
            rawViewData.add(new Tuple2<>(13, 96));
            rawViewData.add(new Tuple2<>(14, 99));
            rawViewData.add(new Tuple2<>(13, 100));

            return jsc.parallelizePairs(rawViewData);
        }

        return jsc.textFile("src/main/resources/udemy/virtualPairProgrammers/viewingFigures/views-*.csv")
                .mapToPair(line -> {
                    String[] columns = line.split(",");
                    return new Tuple2<>(
                            Integer.valueOf(columns[0]),
                            Integer.valueOf(columns[1])
                    );
                });
    }

    private static JavaPairRDD<Integer, String> setUpTitlesDataRdd(JavaSparkContext sc, boolean testMode) {

        if (testMode) {
            // (chapterId, title)
            List<Tuple2<Integer, String>> rawTitles = new ArrayList<>();
            rawTitles.add(new Tuple2<>(1, "How to find a better job"));
            rawTitles.add(new Tuple2<>(2, "Work faster harder smarter until you drop"));
            rawTitles.add(new Tuple2<>(3, "Content Creation is a Mug's Game"));
            return sc.parallelizePairs(rawTitles);
        }
        return sc.textFile("src/main/resources/udemy/virtualPairProgrammers/viewingFigures/titles.csv")
                .mapToPair(commaSeparatedLine -> {
                    String[] cols = commaSeparatedLine.split(",");
                    return new Tuple2<>(Integer.valueOf(cols[0]), cols[1]);
                });
    }
}
