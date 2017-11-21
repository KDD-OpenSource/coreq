/* vim:set ts=8 sw=2 sts=2 noet: */
/* 
   Linked list datatype
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

#include "stdlib.h"
#include "stdio.h"
#include "list.h"

// DOUBLE

int llist_double_init(llist_double *llist) {
  llist->first = NULL;
  llist->last = NULL;
  llist->len = 0;
  return 0;
}

void llist_double_destroy(llist_double *llist) {
  llist_item_double *cur, *prev;
  cur = llist->first;
  while (cur != NULL) {
    prev = cur;
    cur = prev->next;
    free(prev);
  }
  llist->len = 0;
}

void llist_double_print(llist_double *llist) {
  llist_item_double *cur;
  cur = llist->first;
  while (cur != NULL) {
    printf("%.2f ", cur->data);
    cur = cur->next;
  }
  printf("\n");
}

int llist_double_push_back(llist_double *llist, double data) {
  llist_item_double *new = malloc(sizeof(llist_item_double));
  if (new == NULL) {
    return 1;
  }
  new->data = data;
  new->next = NULL;
  new->prev = llist->last;

  if (llist->last == NULL) {
    llist->last = new;
    llist->first = new;
  } else {
    llist->last->next = new;
    llist->last = new;
  }
  llist->len++;

  return 0;
}

long int llist_double_size(llist_double *llist) {
  return llist->len;
}

double llist_double_back(llist_double *llist) {
  return llist->last->data;
}

double llist_double_front(llist_double *llist) {
  return llist->first->data;
}


// UNSIGNED LONG

int llist_ul_init(llist_ul *llist) {
  llist->first = NULL;
  llist->last = NULL;
  llist->len = 0;
  return 0;
}

void llist_ul_destroy(llist_ul *llist) {
  llist_item_ul *cur, *prev;
  cur = llist->first;
  while (cur != NULL) {
    prev = cur;
    cur = prev->next;
    free(prev);
  }
  llist->len = 0;
}

void llist_ul_print(llist_ul *llist) {
  llist_item_ul *cur;
  cur = llist->first;
  while (cur != NULL) {
    printf("%lu ", cur->data);
    cur = cur->next;
  }
  printf("\n");
}

int llist_ul_push_back(llist_ul *llist, unsigned long data) {
  llist_item_ul *new = malloc(sizeof(llist_item_ul));
  if (new == NULL) {
    return 1;
  }
  new->data = data;
  new->next = NULL;
  new->prev = llist->last;

  if (llist->last == NULL) {
    llist->last = new;
    llist->first = new;
  } else {
    llist->last->next = new;
    llist->last = new;
  }
  llist->len++;

  return 0;
}

long int llist_ul_size(llist_ul *llist) {
  return llist->len;
}

unsigned long llist_ul_back(llist_ul *llist) {
  return llist->last->data;
}

unsigned long llist_ul_front(llist_ul *llist) {
  return llist->first->data;
}

// remove item from its current list (position) and move to back of another list
int llist_ul_relink(llist_item_ul *item, llist_ul *from, llist_ul *to) {
  llist_item_ul *pred;
  llist_item_ul *succ;

  if ((item == NULL) || (from->len == 0))
    return 1;

  pred = item->prev;
  succ = item->next;

  // append item to the end of 'to'
  item->prev = to->last;
  item->next = NULL;
  if (to->last != NULL) {
    to->last->next = item;
  }
  to->last = item;
  if (to->first == NULL) {
    to->first = item;
  }

  // relink pred and succ in 'from'
  if (pred != NULL) {
    pred->next = succ;
  } else {
    from->first = succ;
  }
  if (succ != NULL) {
    succ->prev = pred;
  } else {
    from->last = pred;
  }

  from->len--;
  to->len++;
  return 0;
}

// append all items from a given list to the end of another list
int llist_ul_relink_all(llist_ul *from, llist_ul *to) {
  if (from->first == NULL) {
    // source is empty
    return 0;
  }
  if (to->last == NULL) {
    // destination is empty
    to->first = from->first;
    to->last = from->last;
    to->len = from->len;
  } else {
    from->first->prev = to->last;
    to->last->next = from->first;
    to->last = from->last;
    to->len += from->len;
  }
  from->first = NULL;
  from->last = NULL;
  from->len = 0;
  return 0;
}


// POINTER

int llist_ptr_init(llist_ptr *llist) {
  llist->first = NULL;
  llist->last = NULL;
  llist->len = 0;
  return 0;
}

void llist_ptr_destroy(llist_ptr *llist) {
  llist_item_ptr *cur, *prev;
  cur = llist->first;
  while (cur != NULL) {
    prev = cur;
    cur = prev->next;
    free(prev);
  }
  llist->len = 0;
}

void llist_ptr_print(llist_ptr *llist) {
  llist_item_ptr *cur;
  cur = llist->first;
  while (cur != NULL) {
    printf("%lx ", (long unsigned int) cur->data);
    cur = cur->next;
  }
  printf("\n");
}

int llist_ptr_push_back(llist_ptr *llist, void *data) {
  llist_item_ptr *new = malloc(sizeof(llist_item_ptr));
  if (new == NULL) {
    return 1;
  }
  new->data = data;
  new->next = NULL;
  new->prev = llist->last;

  if (llist->last == NULL) {
    llist->last = new;
    llist->first = new;
  } else {
    llist->last->next = new;
    llist->last = new;
  }
  llist->len++;

  return 0;
}

long int llist_ptr_size(llist_ptr *llist) {
  return llist->len;
}

void *llist_ptr_back(llist_ptr *llist) {
  return llist->last->data;
}

void *llist_ptr_front(llist_ptr *llist) {
  return llist->first->data;
}
