/* vim:set ts=8 sw=2 sts=2 noet:  */
/* 
   Linked list data type
   Copyright (C) 2017  Erik Scharwaechter <erik.scharwaechter@hpi.de>
   Copyright (C) 2015  Erik Scharwaechter <erik.scharwaechter@rwth-aachen.de>

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 
   02110-1301 USA

*/

/**
 * Linked list data type.
 */

// DOUBLE

typedef struct llist_item_double {
  double data;
  struct llist_item_double *next;
  struct llist_item_double *prev;
} llist_item_double;

typedef struct llist_double {
  llist_item_double *first;
  llist_item_double *last;
  long int len;
} llist_double;

int llist_double_init(llist_double *llist);
void llist_double_destroy(llist_double *llist);
void llist_double_print(llist_double *llist);
int llist_double_push_back(llist_double *llist, double data);
long int llist_double_size(llist_double *llist);
double llist_double_back(llist_double *llist);
double llist_double_front(llist_double *llist);
//int llist_double_relink(llist_item_double *item, llist_double *from, llist_double *to);
//int llist_double_relink_all(llist_double *from, llist_double *to);


// UNSIGNED LONG INT

typedef struct llist_item_ul {
  unsigned long data;
  struct llist_item_ul *next;
  struct llist_item_ul *prev;
} llist_item_ul;

typedef struct llist_ul {
  llist_item_ul *first;
  llist_item_ul *last;
  long int len;
} llist_ul;

int llist_ul_init(llist_ul *llist);
void llist_ul_destroy(llist_ul *llist);
void llist_ul_print(llist_ul *llist);
int llist_ul_push_back(llist_ul *llist, unsigned long data);
long int llist_ul_size(llist_ul *llist);
unsigned long llist_ul_back(llist_ul *llist);
unsigned long llist_ul_front(llist_ul *llist);
int llist_ul_relink(llist_item_ul *item, llist_ul *from, llist_ul *to);
int llist_ul_relink_all(llist_ul *from, llist_ul *to);


// PTR

typedef struct llist_item_ptr {
  void *data;
  struct llist_item_ptr *next;
  struct llist_item_ptr *prev;
} llist_item_ptr;

typedef struct llist_ptr {
  llist_item_ptr *first;
  llist_item_ptr *last;
  long int len;
} llist_ptr;

int llist_ptr_init(llist_ptr *llist);
void llist_ptr_destroy(llist_ptr *llist);
void llist_ptr_print(llist_ptr *llist);
int llist_ptr_push_back(llist_ptr *llist, void *data);
long int llist_ptr_size(llist_ptr *llist);
void *llist_ptr_back(llist_ptr *llist);
void *llist_ptr_front(llist_ptr *llist);
//int llist_ptr_relink(llist_item_ptr *item, llist_ptr *from, llist_ptr *to);
//int llist_ptr_relink_all(llist_ptr *from, llist_ptr *to);

