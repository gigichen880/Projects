/* 
	Given n numbers, q queries
	Return the partial sum for given interval within each query
	For 50% data, n、q<=200
	For 100% data, n、q<=2000
*/

import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		int n = input.nextInt();
		int []a = new int [n];
		for (int i=0; i<a.length; i++){
			a[i] = input.nextInt();
		}
		int q = input.nextInt();
		int [][]b = new int[q][2];
		int sum = 0;
		for (int i=0; i<q; i++){
			for (int j=0; j<2; j++){
				b[i][j] = input.nextInt();
			}
		}
		for (int i=0; i<q; i++){
			for (int j=b[i][0]-1; j<=b[i][1]-1; j++){
				sum+=a[j];
			}
			System.out.println(sum);
			sum = 0;
		}
	}
}