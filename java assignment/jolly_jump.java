/*
A "jolly jump" is a sequence of length n (n>0) 
if and only if the absolute value of the difference between neighboring elements 
is exactly from 1 to (n-1) after sorting

For example, 1 4 2 3 is "jolly jump" because the absolute values of the differences are 3, 2, and 1

Of course, any sequence containing only a single element must be "jolly jump"
Write a program that determines whether a given sequence is "jolly jump"
*/

import java.util.Scanner;
class Main{
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		int n = input.nextInt();
		int []a = new int [n];
		int []b = new int [n];
		int flag = 0;
		for (int i=0;i<n;i++){
			a[i] = input.nextInt();
		}
		for (int i=0;i<n-1;i++){
			if (a[i+1]-a[i]>=0){
				int x = a[i+1]-a[i];
				b[i]=x;
			}
		
			else{
				int x = a[i]-a[i+1];
				b[i]=x;
			}
		}
		
		for (int i=1;i<=n-1;i++){
			for (int j=0;j<n-1;j++){
				if(i==b[j]){
					flag=flag+1;
					break;
				
				}
			}
			if (flag!=i){
				System.out.println("not jolly");
				break;
			}
		}
		if (flag==n-1){
			System.out.println("Jolly");
		}
	}	
}