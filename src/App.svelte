<script>
  import { Router, Link, Route } from 'svelte-routing';
  import Home from './routes/home.svelte';
  import NetworkViewer from './routes/network-viewer.svelte';

  export let url = '';
</script>

<Router {url}>
  <nav>
    <Link to="/">Home</Link>
    <Link to="network-viewer">Network Viewer</Link>
  </nav>
  <div>
    <Route path="network-viewer" component={NetworkViewer} />
    <Route path="/"><Home /></Route>
  </div>
</Router>
