/*
	Inputs are two strings
	Count the length of the strings, the number of spaces, the number of letters and the number of digits 
	Use a function to compare the 4 parameters within 2 structs	
*/
import java.util.Scanner;

public class Main {
	public static int[] Cal(String s)
	{
		int []a = new int[4]; 
		a[0] = s.length();
		for (int i=0; i<s.length(); i++){
			if (s.substring(i, i+1).equals(" "))
				a[1]++;
			else if (s.substring(i, i+1).compareTo("0")>=0 && s.substring(i, i+1).compareTo("9")<=0){
				a[3]++;
			}
			else if (s.substring(i, i+1).compareTo("A")>=0 && s.substring(i, i+1).compareTo("Z")<=0){
				a[2]++;
			}
			else if (s.substring(i, i+1).compareTo("a")>=0 && s.substring(i, i+1).compareTo("z")<=0){
				a[2]++;
			}
		}
			return a;
	}
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		String s = input.nextLine();
		String t = input.nextLine();
		int []a = new int [4];
		int []b = new int [4];
		a = Cal(s);
		b = Cal(t);
		if (a[0]==b[0])
			System.out.println("True");
		else 
			System.out.println("False");
		
		if (a[1]==b[1])
			System.out.println("True");
		else 
			System.out.println("False");
		
		if (a[2]==b[2])
			System.out.println("True");
		else 
			System.out.println("False");
		
		if (a[3]==b[3])
			System.out.println("True");
		else 
			System.out.println("False");
	}
}