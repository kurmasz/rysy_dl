
To layout the graph:  `python cli.pl layout <filename>`

## High-level nice to have:
   * Put all wires in the same vertical channel if they won't overlap.
   * Nudge gates and/or dummy nodes up and down to avoid bends (when possible)


## `example1.pkl`
  * `((d * (a + b + c + d)') * (d + (a + b + c + d)')')'`
  * Input `d` goes directly to layer 1 (hence dummy nodes in layer 2)
  * Both dummy nodes in layer 2 attach to `d` (which means we really don't need both dummy nodes.)
  * Extra "up and down" because of placement of dummy nodes. The first step may be to eliminate the 
    unnecessary dummy nodes, then go from there.
  * Idea: What if we modified the graph so that instead of dummy nodes, we just added "invisible" gates?  


## `example2.pkl`
 * `((d + (c + a' + b'))' + (d ^ (c + a' + b')))`
 * Chains of more than one dummy node (e.g., 5 to 0)
 * Multiple dummy nodes sharing the same vertical channel
 * Wire from input d5 to the vertical channel for dummy nodes clips the NOT gate.
   * This might be fixed automatically if we change how dummy nodes work.
 * Dot missing on shared routes from 6 to 0 and 6 to 1

 ## `example3.pkl`
  * The two gates at level 1 have two inputs each that are attached to the two gates at level 2.
    But, they end up in the same vertical channel. I suspect this is a problem with the
    "distance from middle" metric.