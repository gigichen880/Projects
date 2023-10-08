/* 
	Given a series a of length n and q queries.
	There are two types of queries:
	Query 1: Given x and y, ask for the sum from the xth to the yth number.
	Query 2: Given x and y, ask for the weighted sum from the xth to the yth number. 
	Where the 1st number in the interval has a weight of 1, the 2nd number has a weight of 2, and so on.
	n, q<=100000; each number in the series does not exceed 1000
*/

import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		int n = input.nextInt();
		int []a = new int [n];
		long []sum = new long [n+1];
		long []sum_w = new long [n+1];
		for (int i=0; i<n; i++){
			a[i] = input.nextInt();
		}
		sum[0]=0;
		sum_w[0]=0;
		for (int i=1; i<n+1; i++){
			sum[i]=sum[i-1]+a[i-1];
			sum_w[i]=sum_w[i-1]+a[i-1]*i;
		}
		int q = input.nextInt();
		int k,x,y;
		long c[] = new long [q];
		for (int i=0; i<q; i++){
			k = input.nextInt();
			if (k==1){
				x = input.nextInt();
				y = input.nextInt();
				c[i]=sum[y]-sum[x-1];
			}
			else{
				x = input.nextInt();
				y = input.nextInt();
				c[i]=sum_w[y]-sum_w[x-1]-(x-1)*(sum[y]-sum[x-1]);
			}
		}
		for (int i=0; i<q; i++){
			System.out.println(c[i]);
		}
	}
}