#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string.h>
#include <ctype.h>
#include "phonebook.h"

#define MAX_SIZE 100     // size of array to hold our contacts

/* array to hold our contacts */
Contact phonebook[MAX_SIZE];

/* current size of the phonebook array */
int currentSize = 0;

void strtolower(char *str)
{
    int len = strlen(str);
    int i;
    for(i=0; i<len; i++) {
        str[i] = tolower(str[i]);
    }
}


void addEntry()
{
    if(currentSize == MAX_SIZE) {
        puts("Error: phonebook is fulL!");
        return;
    }

    Contact c;
    printf("Enter Name: ");
    gets(c.Name);
    printf("Enter Mobile Number: ");
    gets(c.mobileNumber);
    printf("Enter email_add: ");
    gets(c.email_add);

    phonebook[currentSize] = c;
    ++currentSize;

    printf("\nContact added.\n");
}

void display(int index)
{
    if(index < 0 || index >= currentSize) {
        puts("Error: invalid index!");
        return;
    }

    Contact c = phonebook[index];
    printf("Name: %s\n", c.Name);
    printf("Mobile No : %s\n", c.mobileNumber);
    printf("email_add : %s\n", c.email_add);


}

void displayAll()
{
    if(currentSize == 0) {
        puts("Phonebook is empty!");
        return;
    }
    int i;
    for(i=0; i<currentSize; i++) {
        display(i);
        puts(""); // extra new line
    }
}

void search(char *key, search_t type)
{
    int found = 0;
    int i;

    strtolower(key);
    char content[41];

    if(type == NAME) { // search by  name
        for(i=0; i<currentSize; i++) {

            strcpy(content, phonebook[i].Name);
            strtolower(content);

            if(strcmp(content, key) == 0) {
                display(i);
                found = 1;
            }
        }
    }
    else if(type  == MOBILE_NUMBER) { // search by mobile number
        for(i=0; i<currentSize; i++) {
            strcpy(content, phonebook[i].mobileNumber);
            strtolower(content);

            if(strcmp(content, key) == 0) {
                display(i);
                found = 1;

            }
        }
    }

    else {
        puts("Error: invalid search type!");
        return;
    }

    if(!found) {
        puts("Not found in the phone book");
    }

}

void removeEntry(char *mobileNumber)
{
    if(currentSize == 0) {
        puts("Phonebook is empty! Nothing to delete!");
        return;
    }


    int i, j;
    int count = 0;
    for(i=0; i<currentSize; i++) {
        if(strcmp(phonebook[i].mobileNumber, mobileNumber) == 0) {
            for(j=i; j<currentSize-1; j++) {
                phonebook[j] = phonebook[j+1];
            }

            strcpy(phonebook[j].Name, "");
            strcpy(phonebook[j].mobileNumber, "");
            strcpy(phonebook[j].email_add, "");

            currentSize -= 1;
            ++count;
        }
    }
    if(count == 0) {
        puts("No entry deleted");
    }
    else {
        printf("%d entries deleted\n", count);
    }
}

void readFromFile()
{
    FILE *fp;
    if((fp = fopen("phonebook.db", "rb")) == NULL) {
        return;
    }


    /* read the size of the phone book */
    if(fread(&currentSize, sizeof(currentSize), 1, fp) != 1) {
        return;
    }

    /* read the actual phone book content */
    if(fread(phonebook, sizeof(phonebook), 1, fp) != 1) {
        return;
    }

}

void saveToFile()
{
    FILE *fp;
    if( (fp = fopen("phonebook.db", "wb")) == NULL ) {
        puts("Error: can't create a database file!");
        return;
    }

    /* Save the current size of the phonebook */
    if (fwrite(&currentSize, sizeof(currentSize), 1, fp) != 1 ) {
        puts("Error: can't save data!");
        return;
    }

    /* save the phonebook contents */
    if(fwrite(phonebook, sizeof(phonebook), 1, fp) != 1) {
        puts("Error: can't save data!");
        return;
    }

    puts("Phonebook saved to file successfully!");
}




void update_info(char *mobileNumber){
    int found = 0;
    int i;

    strtolower(mobileNumber);
    char content[41];


        for(i=0; i<currentSize; i++) {

            strcpy(content, phonebook[i].mobileNumber);
            strtolower(content);

            if(strcmp(content, mobileNumber) == 0) {
                Contact c = phonebook[i];
                printf("Enter Name: ");
                gets(c.Name);
                printf("Enter Mobile Number: ");
                gets(c.mobileNumber);
                printf("Enter email_add: ");
                gets(c.email_add);
                phonebook[i] = c;
                found = 1;
                break;
            }
        }


}
