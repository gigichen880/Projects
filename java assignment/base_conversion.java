// Converts a decimal integer n to a binary number

import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		int n = input.nextInt();
		int []a = new int [500];
		int num = 0;
		while (n!=0){
			a[num] = n%2;
			num++;
			n/=2;
		}
		for (int i = num-1; i>=0; i--){
			System.out.print(a[i]);
		}	
	}
}