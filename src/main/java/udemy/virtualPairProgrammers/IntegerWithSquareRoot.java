package udemy.virtualPairProgrammers;

public class IntegerWithSquareRoot {

    public Integer integer;
    public Double squareRoot;

    IntegerWithSquareRoot(Integer val) {
        this.integer = val;
        this.squareRoot = Math.sqrt(val);
    }
}
