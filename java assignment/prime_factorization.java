/*
	Input n and output n in the form of prime factorization 
	in order from smallest to largest prime factor.
	That is, output n = a1^p1*a2^p2*...ak^pk.

	Note that pi must be at least 1. If pi = 1, (^pi) is ignored.
	Example: 6=2*3	12=2^2*3	19=19
*/
import java.util.Scanner;

public class Main {	
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		int n = input.nextInt();
		int cnt = 0;
		
		System.out.print(n+"=");
		for (int i=2; i<=Math.sqrt(n); i++){
			if (n%i==0){
				System.out.print(i);
			}
			else
				continue;
			while (n%i==0){
				n=n/i;
				cnt++;
			}
			if (cnt>1)
				System.out.print("^"+cnt);
			cnt=0;
			if (n!=1)
				System.out.print("*");
		}
		if (n>1)
			System.out.print(n+" ");
	}
}