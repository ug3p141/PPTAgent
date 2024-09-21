<template>
  <div id="app">
    <header class="app-header">
      <h1>PPTAgent</h1>
    </header>
    <main class="app-main">
      <router-view />
    </main>
    <footer class="app-footer">
      您所上传的所有数据将被分析并被保存，请注意您的数据安全
    </footer>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style scoped>
#app {
  display: flex;
  flex-direction: column;
  height: 100vh; /* Make the app take the full viewport height */
  background-color: aliceblue;
}

.app-header {
  font-family: 'American Typewriter', serif;
  text-align: center;
  font-weight: bold;
  font-size: 20px;
  height: 10vh; /* Fixed height for header */
  color: saddlebrown;
  margin-top: 10vh;
  display: flex;
  align-items: center;
  justify-content: center;
}

.app-main {
  flex: 1; /* Main content takes up the remaining space */
  overflow:hidden; /* Prevent scrolling */
  font-family: "Arial Rounded MT Bold";
}

.app-footer {
  text-align: center;
  font-family: "Times New Roman", Times, serif;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>