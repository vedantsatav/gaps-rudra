#include "System.hpp"
#include <iostream>
int main()
{
 System::profile("queries", [&]() {

		 for (int j=0;j<=10;j++)
		 {
		 std::cout<<"Hello";
		 
		 }
		 });
return 0;
}
