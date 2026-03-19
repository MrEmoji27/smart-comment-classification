import {Composition, Folder} from 'remotion';
import {ProductDemo} from './ProductDemo';

export const RemotionRoot = () => {
  return (
    <Folder name="Product-Demos">
      <Composition
        id="ProductDemo"
        component={ProductDemo}
        durationInFrames={900}
        fps={30}
        width={1920}
        height={1080}
        defaultProps={{
          title: 'Smart Comment Classification',
          subtitle: 'Production-ready NLP for text and batch workflows',
        }}
      />
    </Folder>
  );
};
