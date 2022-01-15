<script lang="ts">
  import { onMount } from 'svelte';
  import * as PIXI from 'pixi.js';
  import MultilayerPerceptron from '../scripts/neuralnet';
  import { randomizeMLP } from '../scripts/factory';

  let elem: HTMLCanvasElement;
  let renderer: PIXI.Renderer;
  let stage: PIXI.Container;

  onMount(() => {
    renderer = new PIXI.Renderer({
      view: elem,
      width: 800,
      height: 600,
    });

    stage = new PIXI.Container();
  });

  function createNetwork() {
    const mlp = new MultilayerPerceptron(5, 3, 2, 4, (x) => x);

    // randomizeMLP(mlp);

    // mlp.computeNeurons();

    // console.log(mlp);
  }

  createNetwork();
</script>

<div class="w-full h-full">
  <div class="mx-auto w-max">
    <canvas bind:this={elem} />
  </div>
</div>
