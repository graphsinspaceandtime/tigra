# Data bank

A package which is used for loading datasets as `EventSequence` objects.
Here are also descriptions of datasets which might be useful for future work.

## `ia-hospital-ward-proximity-attr`

$\Delta t = 20s$

Tab separated values:

1. id of first node (i)
2. id of second node (j)
3. timestamp
4. status of i (class)
5. stauts of j (class)

```python
event_data = {
  "src_class": int, # status of i
  "dst_class": int, # status of j
}
```

## `ia-primary-school-proximity-attr`

$\Delta t = 20s$

Tab separated values:

1. id of first node (i)
2. id of second node (j)
3. timestamp
4. class of i
5. class of j

```python
event_data = {
  "src_class": str, # class of i
  "dst_class": str, # class of j
}
```
