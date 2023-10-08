/* 
	Given n numbers, q queries
	Return the partial sum for given interval within each query
	For 30% data, n、q<=200
	For 60% data，n、q<=2000
	For 100% data，n、q<=200000
*/

import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		int n = input.nextInt();
		int []a = new int [n];
		int []sum = new int [n+1];
		for (int i=0; i<n; i++){
			a[i] = input.nextInt();
		}
		sum[0]=0;
		for (int i=1; i<n+1; i++){
			sum[i]=sum[i-1]+a[i-1];
		}
		int q = input.nextInt();
		int x,y;
		int c[] = new int [q];
		for (int i=0; i<q; i++){
			x = input.nextInt();
			y = input.nextInt();
			c[i]=sum[y]-sum[x-1];
		}
		for (int i=0; i<q; i++){
			System.out.println(c[i]);
		}
	}
}