import org.apache.spark.sql.api.java.UDF1;

import java.text.SimpleDateFormat;
import java.util.Map;
import java.util.TimeZone;
import java.util.concurrent.ConcurrentHashMap;

public class TimeEpoch {

    public static transient ThreadLocal<SimpleDateFormat> dateFormatThreadLocal;
    private SimpleDateFormat simpleDateFormat;

    public static Map<String, Long> timeMap = new ConcurrentHashMap<>();
    public static Map<String, Integer> fileNameCollectorIdMap = new ConcurrentHashMap<>();

    public static void main(String[] args) throws Exception {
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyyMMddHHmmss");
        simpleDateFormat.setTimeZone(TimeZone.getTimeZone("Asia/Kolkata"));
        System.out.println(new TimeInEpoch_1(simpleDateFormat).call("20240716230808"));
    }

    public static class TimeInEpoch_1 {
        public transient ThreadLocal<SimpleDateFormat> dateFormatThreadLocal;
        private SimpleDateFormat simpleDateFormat;

        public TimeInEpoch_1(SimpleDateFormat simpleDateFormat) {
            this.simpleDateFormat = simpleDateFormat;
        }

        public Long call(String time) throws Exception {
            if (time == null) {
                return null;
            }
            if (dateFormatThreadLocal == null) {
                synchronized (TimeInEpoch_1.class) {
                    if (dateFormatThreadLocal == null) {
                        this.dateFormatThreadLocal = new ThreadLocal() {
                            protected SimpleDateFormat initialValue() {
                                Object clone = simpleDateFormat.clone();
                                SimpleDateFormat dateFormat = (SimpleDateFormat) clone;
                                return dateFormat;
                            }
                        };
                    }
                }
            }
            Long timeInEpoch = timeMap.get(time);
            if (timeInEpoch == null) {
                try {
                    timeInEpoch = dateFormatThreadLocal.get().parse(time).getTime();
                } catch (Exception e) {
                    System.out.println("unable to parse time :" + time);
                    return null;
                }
                timeMap.put(time, timeInEpoch);
            }
            return (timeInEpoch / 1000);
        /*long timeInEpoch = 0l;
        try {
            timeInEpoch = simpleDateFormat.parse(time).getTime();
        } catch (Exception e) {
            System.out.println("unable to parse time :" + time);
            throw e;
        }
        return (timeInEpoch / 1000 );*/
        }
    }
}
