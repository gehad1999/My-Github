#ifndef PHONEBOOK_H
#define PHONEBOOK_H

/* structure to hold the informations of a single contact */
typedef struct {
    char Name[100];
    char mobileNumber[20];
    char email_add[100];
} Contact;

/* enumerator for different search types */
typedef enum { NAME ,  MOBILE_NUMBER } search_t;

/* Function Prototypes */
void addEntry();
void display(int index);
void displayAll();
void search(char *key, search_t type);
void removeEntry(char *mobileNumber);
void readFromFile();
void saveToFile();
void update_info();
void strtolower(char *str);


#endif // PHONEBOOK_H
